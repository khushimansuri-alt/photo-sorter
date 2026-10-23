"""
School Photo Sorter Dashboard
-----------------------------
Paste a Google Drive folder link and the AI sorts the photos by event.

Install (one time):
    py -3.12 -m pip install streamlit torch transformers pillow requests google-api-python-client google-auth-httplib2 google-auth-oauthlib

Run:
    py -3.12 -m streamlit run photo_dashboard.py

To connect to Google Drive, credentials.json must be in the same folder as this file.
"""

import csv
import datetime
import hmac
import io
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import streamlit as st
import torch
import torch.nn.functional as F
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import build
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

BASE = Path(__file__).resolve().parent
CRED_FILE = BASE / "credentials.json"
TOKEN_FILE = BASE / "token.json"


def get_secret(name, default=None):
    """Read a value from Streamlit secrets (used when the app is deployed online)."""
    try:
        return st.secrets[name]
    except Exception:
        return default


OAUTH_CFG = get_secret("google_oauth")
APP_PASSWORD = get_secret("app_password")
CLOUD_MODE = OAUTH_CFG is not None  # True when the app is deployed with Google web-login secrets

# Previews are cached on disk. When deployed online, use a temporary folder.
CACHE = (Path(tempfile.gettempdir()) / "thumb_cache") if CLOUD_MODE else (BASE / "thumb_cache")

SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"
MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PER_PAGE = 40
UNSURE = "Check_Manually"

SOURCE_DRIVE = "Google Drive link"
SOURCE_LOCAL = "Folder on this computer"

DEFAULT_CATEGORIES = """\
Independence_Day | a photo of students hoisting the Indian tricolor flag; children dressed in saffron white and green; students saluting the Indian national flag
Republic_Day | a Republic Day parade by school students; students marching with the Indian flag
Annual_Function | students performing on a stage during an annual function; a school cultural program with dance on stage; children in colorful costumes performing a dance
Sports_Day | school children running a race on a sports ground; sports day with students playing games; students receiving medals and trophies
Prize_Distribution | a student receiving a certificate or trophy from a guest; prize distribution ceremony on stage
Teachers_Day | students celebrating Teachers Day with their teacher; teachers honored with flowers and gifts
Birthday_Celebration | a birthday cake being cut in a classroom; children celebrating a birthday with balloons
Festival_Diwali_Holi | children celebrating Holi with colors; Diwali celebration with diyas and rangoli
Assembly_Prayer | school morning assembly with students standing in rows
Classroom_Activity | students sitting in a classroom with a teacher; children doing a craft activity in class
Science_Exhibition | students showing science project models at an exhibition
Group_Photo | a posed group photo of many students and teachers
"""

st.set_page_config(page_title="School Photo Sorter", page_icon="📸", layout="wide")


class NeedCredentials(Exception):
    pass


class NeedLogin(NeedCredentials):
    pass


# ------------------------- AI helpers -------------------------
@st.cache_resource(show_spinner="Loading the AI model (it downloads on the first run)...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor, device


def as_tensor(x):
    """Newer transformers versions may return an output object instead of a tensor."""
    if torch.is_tensor(x):
        return x
    if getattr(x, "pooler_output", None) is not None:
        return x.pooler_output
    return x[0]


@st.cache_data(show_spinner=False, max_entries=3000)
def thumbnail(path: str):
    img = Image.open(path)
    img.draft("RGB", (640, 640))  # fast partial decode for large JPEG files
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((320, 320))
    return img


def parse_categories(text: str):
    cats = {}
    for line in text.splitlines():
        if "|" not in line:
            continue
        name, prompts = line.split("|", 1)
        plist = [p.strip() for p in prompts.split(";") if p.strip()]
        if name.strip() and plist:
            cats[name.strip().replace(" ", "_")] = plist
    return cats


def embed_items(items, bar):
    """Run each item's 'thumb' image through the AI. Images that fail to open are skipped."""
    model, processor, device = load_model()
    ok_items, feats = [], []
    batch_imgs, batch_items = [], []

    def flush():
        if not batch_imgs:
            return
        with torch.no_grad():
            inp = processor(images=batch_imgs, return_tensors="pt").to(device)
            f = as_tensor(model.get_image_features(**inp))
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
        ok_items.extend(batch_items)
        batch_imgs.clear()
        batch_items.clear()

    for i, it in enumerate(items):
        try:
            img = Image.open(it["thumb"])
            img.draft("RGB", (640, 640))
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception:
            continue
        batch_imgs.append(img)
        batch_items.append(it)
        if len(batch_imgs) == 16:
            flush()
        if i % 16 == 0:
            bar.progress(min((i + 1) / len(items), 1.0), text=f"AI is analyzing photos: {i + 1}/{len(items)}")
    flush()
    bar.progress(1.0, text="Scan complete")
    return ok_items, (torch.cat(feats) if feats else torch.empty(0, 512))


def text_embeddings(cats: dict):
    model, processor, device = load_model()
    out = []
    with torch.no_grad():
        for prompts in cats.values():
            inp = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            f = as_tensor(model.get_text_features(**inp))
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(dim=0)
            out.append((f / f.norm()).cpu())
    return torch.stack(out)


def classify(feats, cats, threshold, smooth=0):
    names = list(cats.keys())
    t = text_embeddings(cats)
    probs = (100.0 * feats @ t.T).softmax(dim=-1)
    if smooth > 0:
        # Photos taken one after another (DSC03497, DSC03498, ...) usually belong
        # to the same event, so average the scores over neighboring photos.
        k = 2 * smooth + 1
        x = probs.T.unsqueeze(0)
        x = F.avg_pool1d(x, kernel_size=k, stride=1, padding=smooth, count_include_pad=False)
        probs = x.squeeze(0).T
    conf, idx = probs.max(dim=1)
    labels = [names[int(i)] if float(c) >= threshold else UNSURE for i, c in zip(idx, conf)]
    return labels, [float(c) for c in conf]


# ------------------------- Google Drive helpers -------------------------
def get_creds():
    if CLOUD_MODE:
        creds = st.session_state.get("creds")
        if creds is None:
            raise NeedLogin()
        if not creds.valid and creds.refresh_token:
            creds.refresh(Request())
        return creds
    creds = None
    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        except Exception:
            creds = None
    if creds and creds.valid:
        return creds
    if creds and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_FILE.write_text(creds.to_json())
            return creds
        except Exception:
            pass
    if not CRED_FILE.exists():
        raise NeedCredentials()
    flow = InstalledAppFlow.from_client_secrets_file(str(CRED_FILE), SCOPES)
    creds = flow.run_local_server(port=0)  # opens the Google login page in the browser
    TOKEN_FILE.write_text(creds.to_json())
    return creds


def password_gate():
    """Optional shared password. Active only when 'app_password' is set in the app secrets."""
    if not APP_PASSWORD or st.session_state.get("authed"):
        return
    st.title("📸 School Photo Sorter")
    pw = st.text_input("Password", type="password")
    if pw:
        if hmac.compare_digest(pw.encode(), str(APP_PASSWORD).encode()):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


def make_flow():
    cfg = OAUTH_CFG
    client_config = {
        "web": {
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [cfg["redirect_uri"]],
        }
    }
    try:
        # No PKCE: the Google redirect starts a new browser session, so a code verifier could not be kept.
        return Flow.from_client_config(
            client_config,
            scopes=SCOPES,
            redirect_uri=cfg["redirect_uri"],
            autogenerate_code_verifier=False,
        )
    except TypeError:
        return Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=cfg["redirect_uri"])


def cloud_login_gate():
    """Online mode: every visitor signs in with their own Google account."""
    if st.session_state.get("creds") is not None:
        return
    code = st.query_params.get("code")
    if code:
        ok = False
        try:
            flow = make_flow()
            flow.fetch_token(code=code)
            st.session_state["creds"] = flow.credentials
            ok = True
        except Exception as e:
            st.error(f"Google sign-in failed: {e}")
        st.query_params.clear()
        if ok:
            st.rerun()
    flow = make_flow()
    auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent")
    st.title("📸 School Photo Sorter")
    st.write("Sign in with the Google account that can open the photo folder.")
    st.link_button("Sign in with Google", auth_url)
    st.stop()


def extract_folder_id(text: str):
    text = (text or "").strip()
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", text) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", text)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{15,}", text):
        return text
    return None


def list_drive_images(service, root_id, status):
    images, queue = [], [root_id]
    while queue:
        fid = queue.pop(0)
        token = None
        while True:
            r = service.files().list(
                q=f"'{fid}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, mimeType, thumbnailLink)",
                pageSize=1000,
                pageToken=token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            for f in r.get("files", []):
                if f["mimeType"] == FOLDER_MIME:
                    queue.append(f["id"])
                elif f["mimeType"].startswith("image/"):
                    images.append(f)
            status.text(f"Reading folder... {len(images)} photos found")
            token = r.get("nextPageToken")
            if not token:
                break
    images.sort(key=lambda f: f["name"])
    return images


def fetch_thumb(f, access_token):
    """Fetch a small preview from Drive (not the full multi-megabyte photo)."""
    out = CACHE / f"{f['id']}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        data = None
        link = f.get("thumbnailLink")
        if link:
            link = re.sub(r"=s\d+$", "=s512", link)
            r = requests.get(link, headers=headers, timeout=60)
            if r.status_code == 200 and r.content:
                data = r.content
        if data is None:  # no preview available, so download the full photo instead
            url = f"https://www.googleapis.com/drive/v3/files/{f['id']}?alt=media&supportsAllDrives=true"
            r = requests.get(url, headers=headers, timeout=300)
            if r.status_code != 200:
                return None
            data = r.content
        img = Image.open(io.BytesIO(data))
        img.draft("RGB", (640, 640))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((512, 512))
        img.save(out, "JPEG", quality=85)
        return str(out)
    except Exception:
        return None


def scan_drive(link, status, bar):
    folder_id = extract_folder_id(link)
    if not folder_id:
        raise ValueError("This does not look like a valid Google Drive folder link.")
    creds = get_creds()
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    files = list_drive_images(service, folder_id, status)
    if not files:
        raise ValueError("No photos were found in this folder. Check the link and your access permission.")
    CACHE.mkdir(exist_ok=True)
    token = creds.token
    thumbs = []
    with ThreadPoolExecutor(8) as ex:
        for i, t in enumerate(ex.map(lambda f: fetch_thumb(f, token), files)):
            thumbs.append(t)
            bar.progress((i + 1) / len(files), text=f"Fetching previews: {i + 1}/{len(files)}")
    status.empty()
    return [
        {"key": f["id"], "name": f["name"], "thumb": t, "drive_id": f["id"]}
        for f, t in zip(files, thumbs)
        if t
    ]


def scan_local(folder):
    files = sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return [{"key": p, "name": Path(p).name, "thumb": p, "drive_id": None} for p in files]


def export_to_drive(items, labels, bar):
    """Create a 'Sorted Photos' folder in your own Drive containing shortcuts (no copies)."""
    creds = get_creds()
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H%M")
    root = service.files().create(
        body={"name": f"Sorted Photos {stamp}", "mimeType": FOLDER_MIME}, fields="id"
    ).execute()
    folder_ids = {}
    for lab in sorted(set(labels)):
        r = service.files().create(
            body={"name": lab, "mimeType": FOLDER_MIME, "parents": [root["id"]]}, fields="id"
        ).execute()
        folder_ids[lab] = r["id"]

    bodies = [
        {
            "name": it["name"],
            "mimeType": "application/vnd.google-apps.shortcut",
            "parents": [folder_ids[lab]],
            "shortcutDetails": {"targetId": it["drive_id"]},
        }
        for it, lab in zip(items, labels)
    ]
    errors = [0]

    def cb(request_id, response, exception):
        if exception is not None:
            errors[0] += 1

    step = 25
    for i in range(0, len(bodies), step):
        batch = service.new_batch_http_request(callback=cb)
        for b in bodies[i:i + step]:
            batch.add(service.files().create(body=b, fields="id", supportsAllDrives=True))
        batch.execute()
        done = min(i + step, len(bodies))
        bar.progress(done / len(bodies), text=f"Creating shortcuts in Drive: {done}/{len(bodies)}")
    return root["id"], errors[0]


def unique_dest(folder: Path, name: str):
    dest = folder / name
    stem, suffix = Path(name).stem, Path(name).suffix
    n = 1
    while dest.exists():
        dest = folder / f"{stem}_{n}{suffix}"
        n += 1
    return dest


def download_originals(items, labels, out_dir, bar):
    errors = 0
    for i, (it, lab) in enumerate(zip(items, labels)):
        creds = get_creds()  # refreshes the access token automatically when it expires
        headers = {"Authorization": f"Bearer {creds.token}"}
        d = Path(out_dir) / lab
        d.mkdir(parents=True, exist_ok=True)
        dest = unique_dest(d, it["name"])
        url = f"https://www.googleapis.com/drive/v3/files/{it['drive_id']}?alt=media&supportsAllDrives=true"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=300) as r:
                if r.status_code != 200:
                    errors += 1
                    continue
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
        except Exception:
            errors += 1
        bar.progress((i + 1) / len(items), text=f"Downloading: {i + 1}/{len(items)}")
    return errors


# ------------------------- state -------------------------
ss = st.session_state

# ------------------------- access control (online only) -------------------------
password_gate()
if CLOUD_MODE:
    cloud_login_gate()

ss.setdefault("photo_items", [])
ss.setdefault("feats", None)
ss.setdefault("overrides", {})
ss.setdefault("page", 0)
ss.setdefault("source_mode", None)

# ------------------------- sidebar -------------------------
with st.sidebar:
    st.title("📸 Photo Sorter")
    if CLOUD_MODE:
        mode = SOURCE_DRIVE
        if st.button("Sign out"):
            for k in ("creds", "photo_items", "feats"):
                ss.pop(k, None)
            st.rerun()
    else:
        mode = st.radio("Where are the photos?", [SOURCE_DRIVE, SOURCE_LOCAL])
    if mode == SOURCE_DRIVE:
        link = st.text_input("Google Drive folder link", placeholder="https://drive.google.com/drive/folders/...")
        local_path = ""
    else:
        local_path = st.text_input("Folder path", placeholder=r"C:\Photos\AllPhotos")
        link = ""

    threshold = st.slider("Strictness (lower = more photos get sorted)", 0.20, 0.80, 0.40, 0.05)
    smooth = st.slider(
        "Neighbor smoothing (how many nearby photos to consider)",
        0, 10, 3,
        help="Photos taken in sequence (DSC03497, DSC03498, ...) usually belong to the same event. "
             "0 turns this off. Use a low value if events are mixed together in the folder.",
    )

    with st.expander("Event categories (editable)"):
        st.caption("Format: Name | description 1; description 2  (descriptions in English)")
        cat_text = st.text_area("Categories", DEFAULT_CATEGORIES, height=320, label_visibility="collapsed")

    if st.button("🔍 Scan and sort", type="primary", use_container_width=True):
        status = st.empty()
        bar = st.progress(0.0, text="Starting...")
        try:
            if mode == SOURCE_DRIVE:
                items = scan_drive(link, status, bar)
            else:
                if not local_path or not Path(local_path).is_dir():
                    raise ValueError("Please enter a valid folder path.")
                items = scan_local(local_path)
                if not items:
                    raise ValueError("No photos were found in this folder.")
            ss.photo_items, ss.feats = embed_items(items, bar)
            ss.overrides = {}
            ss.page = 0
            ss.source_mode = mode
            st.success(f"{len(ss.photo_items)} photos scanned.")
        except NeedLogin:
            bar.empty()
            st.error("Your Google session expired. Refresh the page and sign in again.")
        except NeedCredentials:
            bar.empty()
            st.error(
                "credentials.json was not found. Create an OAuth client (Desktop app) in Google Cloud, "
                f"download the JSON file, rename it to credentials.json and place it in: {BASE}"
            )
        except Exception as e:
            bar.empty()
            st.error(f"Error: {e}")

# ------------------------- main -------------------------
st.header("School Photo Sorter Dashboard")

if ss.feats is None or len(ss.photo_items) == 0:
    st.info("Paste a Google Drive folder link in the sidebar and click **Scan and sort**.")
    st.stop()

cats = parse_categories(cat_text)
if not cats:
    st.error("At least one category is required.")
    st.stop()

items = ss.photo_items
auto_labels, confs = classify(ss.feats, cats, threshold, smooth)
final = [ss.overrides.get(it["key"], lab) for it, lab in zip(items, auto_labels)]
all_labels = list(cats.keys()) + [UNSURE]

counts = {k: final.count(k) for k in all_labels}
st.subheader("Summary")
c = st.columns(4)
c[0].metric("Total photos", len(final))
c[1].metric("Sorted automatically", len(final) - counts[UNSURE])
c[2].metric("Need manual check", counts[UNSURE])
c[3].metric("Changed by you", len(ss.overrides))
st.bar_chart({k: v for k, v in counts.items() if v > 0}, horizontal=True)

st.subheader("Review and correct")
choice = st.selectbox(
    "Choose a category",
    all_labels,
    format_func=lambda k: f"{k}  ({counts[k]})",
    on_change=lambda: ss.update(page=0),
)
view = [(it, cf) for it, lab, cf in zip(items, final, confs) if lab == choice]

if not view:
    st.write("There are no photos in this category.")
else:
    pages = max(1, (len(view) + PER_PAGE - 1) // PER_PAGE)
    ss.page = min(ss.page, pages - 1)

    nav = st.columns([1, 1, 6])
    if nav[0].button("⬅ Previous", disabled=ss.page == 0):
        ss.page -= 1
        st.rerun()
    if nav[1].button("Next ➡", disabled=ss.page >= pages - 1):
        ss.page += 1
        st.rerun()
    nav[2].write(f"Page {ss.page + 1} of {pages}")

    chunk = view[ss.page * PER_PAGE:(ss.page + 1) * PER_PAGE]
    grid = st.columns(5)
    for i, (it, cf) in enumerate(chunk):
        with grid[i % 5]:
            try:
                st.image(thumbnail(it["thumb"]), use_container_width=True)
            except Exception:
                st.write("⚠ Could not open this photo")
            st.checkbox(f"{it['name'][:18]} · {cf:.0%}", key=f"sel_{it['key']}")

    selected = [it["key"] for it, _ in chunk if ss.get(f"sel_{it['key']}")]

    def apply_move(target, sel_keys):
        for k in sel_keys:
            ss.overrides[k] = target
            ss[f"sel_{k}"] = False

    st.divider()
    m1, m2 = st.columns([3, 1])
    target = m1.selectbox("Move the selected photos to this category", all_labels, key="move_target")
    m2.write("")
    m2.write("")
    m2.button(f"Move ({len(selected)})", disabled=not selected, on_click=apply_move, args=(target, selected))

# ------------------------- export -------------------------
st.divider()
st.subheader("Export")

if ss.source_mode == SOURCE_DRIVE:
    st.caption(
        "Option 1 creates folders in your own Drive that contain shortcuts to the original photos "
        "(no extra storage is used and the original folder is not changed)."
        + ("" if CLOUD_MODE else " Option 2 downloads the original photos to this computer (needs enough free disk space).")
    )
    st.markdown("**Option 1: Sorted folders in Google Drive**")
    if st.button("☁ Create folders in Drive", type="primary"):
        try:
            bar = st.progress(0.0)
            root_id, errs = export_to_drive(items, final, bar)
            st.success("Done! The folder was created in your My Drive.")
            st.markdown(f"[Open in Drive](https://drive.google.com/drive/folders/{root_id})")
            if errs:
                st.warning(f"{errs} shortcuts could not be created. You can try again.")
        except NeedCredentials:
            st.error("credentials.json was not found.")
        except Exception as e:
            st.error(f"Error: {e}")

    if not CLOUD_MODE:
        st.markdown("**Option 2: Download the original photos to this computer**")
        out_dir = st.text_input("Download location", placeholder=r"D:\Sorted", key="dl_dir")
        if st.button("💾 Download to this computer"):
            if not out_dir:
                st.error("Please enter a download folder path first.")
            else:
                try:
                    bar = st.progress(0.0)
                    errs = download_originals(items, final, out_dir, bar)
                    st.success(f"Done! The photos are in: {out_dir}")
                    if errs:
                        st.warning(f"{errs} photos could not be downloaded (possibly a permission issue).")
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    out_dir = st.text_input("Where should the sorted photos be saved?", placeholder=r"D:\Sorted")
    if st.button("📁 Copy into folders", type="primary"):
        if not out_dir:
            st.error("Please enter an output folder path.")
        else:
            bar = st.progress(0.0)
            for i, (it, lab) in enumerate(zip(items, final)):
                d = Path(out_dir) / lab
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(it["thumb"], unique_dest(d, it["name"]))
                bar.progress((i + 1) / len(items))
            st.success(f"Done! The photos are in: {out_dir}")

buf = io.StringIO()
w = csv.writer(buf)
w.writerow(["photo", "category", "confidence", "manually_changed"])
for it, lab, cf in zip(items, final, confs):
    w.writerow([it["name"], lab, f"{cf:.2f}", "yes" if it["key"] in ss.overrides else "no"])
st.download_button("⬇ Download report (CSV)", buf.getvalue(), "report.csv", "text/csv")
