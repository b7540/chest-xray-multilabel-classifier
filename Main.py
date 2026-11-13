import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import time
import io
from PIL import Image

# --- 1. SAFE IMPORTS ---
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset as TorchDataset, IterableDataset as TorchIterableDataset, DataLoader as TorchDataLoader, random_split, Subset
    from torchvision import transforms, models
    torch.backends.cudnn.enabled = False
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score
    import matplotlib.pyplot as plt
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# FIX: the old import block put `load_dataset` and `login` in one try, so any failure
# disabled both. They are now guarded separately and used only when available.
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except Exception:
    HF_DATASETS_AVAILABLE = False

try:
    from huggingface_hub import login, hf_hub_download
    HF_HUB_AVAILABLE = True
except Exception:
    HF_HUB_AVAILABLE = False

HF_AVAILABLE = HF_DATASETS_AVAILABLE and HF_HUB_AVAILABLE

# Option 2: WebDataset import
try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
except ImportError:
    WEBDATASET_AVAILABLE = False

# Option 3: MosaicML Streaming import
try:
    from streaming import StreamingDataset
    MOSAIC_AVAILABLE = True
except ImportError:
    MOSAIC_AVAILABLE = False


# --- 2. CONFIGURATION & CONSTANTS ---
st.set_page_config(page_title="Medical AI Platform", page_icon="🩻", layout="wide")

DEFAULT_DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia'
]

# Number of streamed rows reserved for validation (HF streaming mode).
# Train stream = everything AFTER these rows, so train and val never overlap.
HF_VAL_SIZE = 200

# Admin password: set ADMIN_PASSWORD as an environment variable or in .streamlit/secrets.toml.
# The fallback keeps the old behaviour, but change it before any real use.
def get_admin_password():
    try:
        if "ADMIN_PASSWORD" in st.secrets:
            return st.secrets["ADMIN_PASSWORD"]
    except Exception:
        pass
    return os.getenv("ADMIN_PASSWORD", "admin123")


# --- 3. HUGGING FACE HELPERS (STREAMING ONLY, NOTHING IS DOWNLOADED IN FULL) ---
def get_hf_stream(repo_id, config_name=""):
    """Always streaming=True so the dataset is never stored on disk."""
    if not HF_DATASETS_AVAILABLE:
        raise RuntimeError("`datasets` library is not available in this environment.")
    config_name = (config_name or "").strip()
    if config_name:
        return load_dataset(repo_id, name=config_name, split="train", streaming=True)
    return load_dataset(repo_id, split="train", streaming=True)


def fetch_hf_preview(repo_id, config=None, split="train", n=4):
    """
    Fetch a few preview rows through Hugging Face's public datasets-server REST API.
    Needs only `requests`: no `datasets` library, no local storage.
    """
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("`requests` is not installed.")
    base = "https://datasets-server.huggingface.co"
    if not config:
        r = requests.get(f"{base}/splits", params={"dataset": repo_id}, timeout=30)
        r.raise_for_status()
        splits = r.json()["splits"]
        match = next((s for s in splits if s["split"] == split), splits[0])
        config, split = match["config"], match["split"]
    r = requests.get(
        f"{base}/first-rows",
        params={"dataset": repo_id, "config": config, "split": split},
        timeout=60
    )
    r.raise_for_status()
    data = r.json()
    return [item["row"] for item in data["rows"][:n]]


def to_pil_image(img_obj):
    """Convert the different image formats HF datasets can return into an RGB PIL image."""
    blank = Image.new('RGB', (224, 224), (0, 0, 0))
    if img_obj is None:
        return blank
    try:
        if isinstance(img_obj, Image.Image):
            return img_obj.convert('RGB')
        if isinstance(img_obj, dict):
            if img_obj.get('bytes'):
                return Image.open(io.BytesIO(img_obj['bytes'])).convert('RGB')
            if img_obj.get('src') and REQUESTS_AVAILABLE:  # datasets-server REST format
                resp = requests.get(img_obj['src'], timeout=30)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert('RGB')
        if isinstance(img_obj, (bytes, bytearray)):
            return Image.open(io.BytesIO(img_obj)).convert('RGB')
    except Exception:
        return blank
    return blank


# --- 4. MODEL FACTORY & GLOBAL CACHING ---
def build_model(model_name="DenseNet121", num_classes=14, pretrained=True, freeze_backbone=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "DenseNet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        target_layer = "denseblock4"

    elif model_name == "EfficientNet-B0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        target_layer = "8"

    elif model_name == "MobileNetV3-Large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[0].in_features
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        target_layer = "16"

    return model.to(device), target_layer

@st.cache_resource
def load_shared_model(weights_path="shared_weights.pth"):
    if not os.path.exists(weights_path):
        return None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        active_classes = st.session_state.get('active_classes', DEFAULT_DISEASE_CLASSES)
        if isinstance(checkpoint, dict) and 'arch' in checkpoint:
            arch = checkpoint['arch']
            num_classes = checkpoint.get('num_classes', len(active_classes))
            state_dict = checkpoint['state_dict']
        else:
            arch = "DenseNet121"
            num_classes = len(active_classes)
            state_dict = checkpoint

        model, default_layer = build_model(arch, num_classes=num_classes, pretrained=False, freeze_backbone=False)
        model.load_state_dict(state_dict)
        model.eval()
        return model, arch, default_layer
    except Exception as e:
        st.error(f"Error loading master model: {e}")
        return None, None, None

# --- 5. DATASET HANDLERS ---
if TORCH_AVAILABLE:
    # 0. Local Dataset Handler
    class PyTorchChestXrayDataset(TorchDataset):
        def __init__(self, dataframe, image_dir, classes=None, is_train=True):
            self.df = dataframe.dropna(subset=['image_id']).reset_index(drop=True) if dataframe is not None else pd.DataFrame()
            self.image_dir = image_dir
            self.classes = classes if classes else DEFAULT_DISEASE_CLASSES
            self.unique_images = self.df['image_id'].unique() if not self.df.empty else []
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            if is_train:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize
                ])

        def __len__(self):
            return len(self.unique_images)

        def __getitem__(self, idx):
            img_id = self.unique_images[idx]
            img_path = os.path.join(self.image_dir, f"{img_id}.dicom")

            if PYDICOM_AVAILABLE and os.path.exists(img_path):
                dicom = pydicom.dcmread(img_path)
                image = dicom.pixel_array
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                alt_path = os.path.join(self.image_dir, f"{img_id}.png")
                if os.path.exists(alt_path):
                    image = np.array(Image.open(alt_path).convert('RGB'))
                else:
                    image = np.zeros((224, 224, 3), dtype=np.uint8)

            image_tensor = self.transform(image)

            annotations = self.df[self.df['image_id'] == img_id]
            label = torch.zeros(len(self.classes), dtype=torch.float32)
            for _, row in annotations.iterrows():
                c_name = row.get('class_name', '')
                if c_name in self.classes:
                    label[self.classes.index(c_name)] = 1.0

            return image_tensor, label

    # OPTION 1: Hugging Face IterableDataset Stream Handler
    class HFStreamingChestXrayDataset(TorchIterableDataset):
        def __init__(self, hf_iterable_dataset, classes=None, is_train=True):
            self.hf_dataset = hf_iterable_dataset
            self.classes = classes if classes else DEFAULT_DISEASE_CLASSES
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            if is_train:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize
                ])

        def __iter__(self):
            num_classes = len(self.classes)
            for row in self.hf_dataset:
                # 1. Image Extraction
                img_obj = row.get('image', row.get('img', row.get('xray', None)))
                image = to_pil_image(img_obj)

                image_tensor = self.transform(image)
                label = torch.zeros(num_classes, dtype=torch.float32)

                # 2. Dynamic Label Extraction
                hf_labels = row.get('labels', row.get('label', row.get('target', None)))
                if hf_labels is not None:
                    if isinstance(hf_labels, list):
                        for l in hf_labels:
                            if isinstance(l, (int, float, np.integer)) and int(l) < num_classes:
                                label[int(l)] = 1.0
                            elif isinstance(l, str) and l in self.classes:
                                label[self.classes.index(l)] = 1.0
                    elif isinstance(hf_labels, (int, float, np.integer)):
                        if int(hf_labels) < num_classes:
                            label[int(hf_labels)] = 1.0
                    elif isinstance(hf_labels, str) and hf_labels in self.classes:
                        label[self.classes.index(hf_labels)] = 1.0

                yield image_tensor, label

    # OPTION 2: WebDataset (.tar Shards) Builder
    def create_webdataset_pipeline(url_pattern, classes=None, is_train=True):
        active_classes = classes if classes else DEFAULT_DISEASE_CLASSES
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if is_train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

        def process_sample(sample):
            # FIX: after wds .decode("pil") the keys are plain extensions ("png", "jpg"),
            # not ".png", so the old endswith('.png') check never matched.
            img_key = [k for k in sample.keys() if k.lower() in ('png', 'jpg', 'jpeg', 'webp', 'img', 'image')]
            if img_key:
                image_data = sample[img_key[0]]
                if isinstance(image_data, Image.Image):
                    image = image_data.convert('RGB')
                else:
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                image = Image.new('RGB', (224, 224), (0, 0, 0))

            image_tensor = transform(image)

            label = torch.zeros(len(active_classes), dtype=torch.float32)
            if 'json' in sample:
                meta = sample['json']
                labels_list = meta.get('labels', meta.get('class_ids', []))
                for l in labels_list:
                    if int(l) < len(active_classes):
                        label[int(l)] = 1.0

            return image_tensor, label

        dataset = (
            wds.WebDataset(url_pattern, handler=wds.warn_and_continue)
            .decode("pil", handler=wds.warn_and_continue)
            .map(process_sample)
        )
        return dataset

    # OPTION 3: MosaicML Streaming Dataset Handler
    if MOSAIC_AVAILABLE:
        class MosaicStreamingChestXray(StreamingDataset):
            def __init__(self, remote_url, local_cache, classes=None, is_train=True):
                self.classes = classes if classes else DEFAULT_DISEASE_CLASSES
                is_local = remote_url is None or not (remote_url.startswith("s3://") or remote_url.startswith("gs://"))
                super().__init__(
                    remote=None if is_local else remote_url,
                    local=remote_url if is_local else local_cache,
                    shuffle=is_train,
                    batch_size=8
                )
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if is_train:
                    self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(degrees=15),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        normalize
                    ])
                else:
                    self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        normalize
                    ])

            def __getitem__(self, idx):
                sample = super().__getitem__(idx)
                if 'image' in sample:
                    raw_img = sample['image']
                    if isinstance(raw_img, bytes):
                        image = Image.open(io.BytesIO(raw_img)).convert('RGB')
                    else:
                        image = Image.fromarray(raw_img).convert('RGB')
                else:
                    image = Image.new('RGB', (224, 224), (0, 0, 0))

                image_tensor = self.transform(image)
                label = torch.zeros(len(self.classes), dtype=torch.float32)
                if 'labels' in sample:
                    lbls = sample['labels']
                    if isinstance(lbls, (list, np.ndarray)):
                        for l in lbls:
                            if int(l) < len(self.classes): label[int(l)] = 1.0
                    elif isinstance(lbls, (int, float)):
                        if int(lbls) < len(self.classes): label[int(lbls)] = 1.0

                return image_tensor, label

@st.cache_data
def load_csv_data(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            header = f.read(10)
            is_zip = header.startswith(b'PK')
            f.seek(128)
            is_dicom = f.read(4) == b'DICM'

        if is_dicom:
            st.error("🚨 **Error:** You accidentally selected a DICOM image instead of the train.csv text file!")
            return None

        if is_zip:
            st.info("📦 ZIP file detected. Extracting CSV on the fly...")
            return pd.read_csv(path, compression='zip')

        try:
            return pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding='latin1')

    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return None

def browse_file():
    if TKINTER_AVAILABLE:
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        fp = filedialog.askopenfilename(master=root, filetypes=[("CSV Files", "*.csv")])
        root.destroy()
        if fp: st.session_state.csv_path = fp

def browse_folder():
    if TKINTER_AVAILABLE:
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        fp = filedialog.askdirectory(master=root)
        root.destroy()
        if fp: st.session_state.train_dir = fp

# --- 6. UI INITIALIZATION & RBAC ---
st.title("🩻 End-to-End Medical AI Platform")
st.sidebar.header("Access Control")
user_role = st.sidebar.radio("Select View:", ["Patient", "Doctor", "Maintenance"])
maintenance_auth = False

if user_role == "Maintenance":
    m_pass = st.sidebar.text_input("Admin Password:", type="password")
    if m_pass == get_admin_password():
        maintenance_auth = True
        st.sidebar.success("Maintenance Unlocked")
    elif m_pass != "":
        st.sidebar.error("Incorrect Password")

if 'csv_path' not in st.session_state:
    st.session_state.csv_path = "train.csv"
if 'train_dir' not in st.session_state:
    st.session_state.train_dir = "train"
if 'data_source' not in st.session_state:
    st.session_state.data_source = "Local Files"
if 'hf_repo' not in st.session_state:
    st.session_state.hf_repo = "timm/nih-chest-xray-14"
if 'hf_config' not in st.session_state:
    st.session_state.hf_config = ""
if 'active_classes' not in st.session_state:
    st.session_state.active_classes = DEFAULT_DISEASE_CLASSES

if user_role == "Maintenance" and maintenance_auth:
    st.sidebar.markdown("---")
    st.sidebar.header("Data Configuration")

    # Safely check for Streamlit secrets without crashing
    try:
        if HF_HUB_AVAILABLE and "HF_TOKEN" in st.secrets:
            login(token=st.secrets["HF_TOKEN"])
            st.sidebar.success("Auto-logged into HF via Secrets!")
    except Exception:
        pass

    # 1. Choose Streaming Engine
    st.session_state.data_source = st.sidebar.radio(
        "Dataset Engine:",
        [
            "Local Files",
            "Hugging Face Cloud Stream (IterableDataset)",
            "WebDataset (.tar HTTP/S3 Shards)",
            "MosaicML Streaming (MDS Format)"
        ]
    )

    # 2. Config per Engine with Dataset Preset Selectors
    if st.session_state.data_source == "Local Files":
        if TKINTER_AVAILABLE:
            st.sidebar.button("Browse for train.csv", on_click=browse_file, use_container_width=True)
        st.session_state.csv_path = st.sidebar.text_input("Path to train.csv:", st.session_state.csv_path)
        if TKINTER_AVAILABLE:
            st.sidebar.button("Browse for train/ folder", on_click=browse_folder, use_container_width=True)
        st.session_state.train_dir = st.sidebar.text_input("Path to train/ folder:", st.session_state.train_dir)

    elif st.session_state.data_source == "Hugging Face Cloud Stream (IterableDataset)":
        if not HF_DATASETS_AVAILABLE:
            st.sidebar.warning("`datasets` is not importable. The inspector will use the REST preview API; training/evaluation need `datasets`.")

        st.sidebar.caption("Streaming only: nothing is downloaded in full, so disk usage stays small.")

        hf_presets = {
            "NIH ChestX-ray14 (Timm)": ("timm/nih-chest-xray-14", ""),
            "Keremberke Chest X-Ray": ("keremberke/chest-xray-classification", "full"),
            "VinBigData Chest X-Ray": ("pulmo/chest-xray-images", ""),
            "Custom Repository": ("custom", "")
        }
        selected_preset = st.sidebar.selectbox("Select Dataset Preset:", list(hf_presets.keys()))

        preset_repo, preset_config = hf_presets[selected_preset]

        if preset_repo == "custom":
            st.session_state.hf_repo = st.sidebar.text_input("Custom HF Repo Path:", st.session_state.hf_repo)
            st.session_state.hf_config = st.sidebar.text_input("Dataset Config / Subset (optional):", st.session_state.hf_config)
        else:
            st.session_state.hf_repo = preset_repo
            st.session_state.hf_config = st.sidebar.text_input("Dataset Config / Subset (optional):", preset_config)
            st.sidebar.caption(f"Selected Repo: `{st.session_state.hf_repo}`")

        hf_token = st.sidebar.text_input("Hugging Face Token (optional):", type="password")
        if st.sidebar.button("Authenticate HF"):
            if not HF_HUB_AVAILABLE:
                st.sidebar.error("`huggingface_hub` is not available.")
            else:
                try:
                    login(token=hf_token)
                    st.sidebar.success("Logged into Hugging Face!")
                except Exception as e:
                    st.sidebar.error(f"HF Login Failed: {e}")

    elif st.session_state.data_source == "WebDataset (.tar HTTP/S3 Shards)":
        if not WEBDATASET_AVAILABLE:
            st.sidebar.error("Please install `webdataset`.")

        wds_presets = {
            "Local Tar Shard Pattern": "shard-{00000..00005}.tar",
            "Custom TAR Shard URL / Path": "custom"
        }
        selected_wds = st.sidebar.selectbox("Select WebDataset Preset:", list(wds_presets.keys()))

        if wds_presets[selected_wds] == "custom":
            st.session_state.wds_url = st.sidebar.text_input("Custom TAR URL/Pattern:", "shard-{00000..00005}.tar")
        else:
            st.session_state.wds_url = wds_presets[selected_wds]
            st.sidebar.caption(f"Shard Pattern: `{st.session_state.wds_url}`")

    elif st.session_state.data_source == "MosaicML Streaming (MDS Format)":
        if not MOSAIC_AVAILABLE:
            st.sidebar.error("Please install `mosaicml-streaming`.")

        mosaic_presets = {
            "Local MDS Directory": "./local_mds_data",
            "S3 Remote Chest X-Ray MDS": "s3://my-medical-bucket/chest-xray-mds",
            "Custom MDS Storage Path": "custom"
        }
        selected_mosaic = st.sidebar.selectbox("Select MDS Store Location:", list(mosaic_presets.keys()))

        if mosaic_presets[selected_mosaic] == "custom":
            st.session_state.mosaic_remote = st.sidebar.text_input("Custom Remote/Local MDS Path:", "./local_mds_data")
        else:
            st.session_state.mosaic_remote = mosaic_presets[selected_mosaic]

        st.session_state.mosaic_cache = st.sidebar.text_input("Local Cache Folder:", "./mosaic_cache")

        with st.sidebar.expander("Cloud / HF Bucket Credentials"):
            aws_key = st.text_input("Access Key ID (Use 'hf' for HF Bucket):", value="hf")
            aws_secret = st.text_input("Secret Key (Your HF Token):", type="password")
            if st.button("Apply Cloud Keys"):
                os.environ["AWS_ACCESS_KEY_ID"] = aws_key
                os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret
                os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
                os.environ["S3_ENDPOINT_URL"] = "https://bucket.hf.co"
                st.success("Cloud credentials set in environment!")

    st.sidebar.markdown("---")
    st.sidebar.header("Upload Pre-trained Model")
    uploaded_model = st.sidebar.file_uploader("Upload .pth file:", type=['pth', 'pt'])
    if uploaded_model is not None:
        with open("shared_weights.pth", "wb") as f:
            f.write(uploaded_model.getbuffer())
        load_shared_model.clear()
        st.sidebar.success("Master model updated globally!")

    if not os.getenv("GEMINI_API_KEY"):
        st.sidebar.markdown("---")
        st.sidebar.header("AI API Configuration")
        manual_key = st.sidebar.text_input("Inject Gemini API Key:", type="password")
        if st.sidebar.button("Save Key"):
            if manual_key:
                os.environ["GEMINI_API_KEY"] = manual_key
                st.sidebar.success("API Key bound!")

active_api_key = os.getenv("GEMINI_API_KEY", "")

# Load CSV only if using local files
df = load_csv_data(st.session_state.csv_path) if st.session_state.data_source == "Local Files" else None

if user_role == "Patient":
    available_tabs = ["Symptom Checker"]
elif user_role == "Doctor":
    available_tabs = ["Symptom Checker", "AI Diagnostics", "Clinical Evaluation"]
elif user_role == "Maintenance" and maintenance_auth:
    available_tabs = ["Symptom Checker", "Dataset Explorer", "AI Training Loop", "AI Diagnostics", "Clinical Evaluation"]
else:
    st.warning("Please enter the maintenance password to access system internals.")
    st.stop()

tabs = st.tabs(available_tabs)
tab_map = dict(zip(available_tabs, tabs))

# --- TAB 1: SYMPTOM CHECKER ---
if "Symptom Checker" in tab_map:
    with tab_map["Symptom Checker"]:
        st.header("Patient Symptom Assessment")
        st.info("Powered by Gemini 2.5 Flash for clinical triage.")
        symptoms = st.text_area("Describe the patient's symptoms (e.g., persistent dry cough, chest tightness for 4 days):")
        if st.button("Analyze Symptoms with AI"):
            if not GENAI_AVAILABLE:
                st.error("google-generativeai library is missing.")
            elif not active_api_key:
                st.warning("Gemini API key missing. Configure in Maintenance tab.")
            elif symptoms:
                with st.spinner("AI is analyzing symptoms and generating triage recommendations..."):
                    try:
                        genai.configure(api_key=active_api_key)
                        llm = genai.GenerativeModel('gemini-2.5-flash')
                        prompt = f"""Act as a clinical triage assistant. Analyze symptoms: "{symptoms}".
Format:
1. **Possible Clinical Categories:**
2. **Recommended Imaging/Scans:**
3. **Triage Priority:**
End with a formal medical disclaimer."""
                        res = llm.generate_content(prompt)
                        st.success("Analysis Complete")
                        st.markdown(res.text)
                    except Exception as e:
                        st.error(f"API Error: {e}")

# --- TAB 2: DATASET EXPLORER & STREAM INSPECTOR ---
if "Dataset Explorer" in tab_map:
    with tab_map["Dataset Explorer"]:
        st.header("Data Inspection & Dataset Explorer")

        if st.session_state.data_source == "Local Files":
            if df is not None and 'class_name' in df.columns:
                st.subheader("Disease Frequency in Local Dataset")
                st.bar_chart(df['class_name'].value_counts())

            if os.path.exists(st.session_state.train_dir) and PYDICOM_AVAILABLE:
                dicom_files = [f for f in os.listdir(st.session_state.train_dir) if f.endswith('.dicom')][:50]
                if dicom_files:
                    sel = st.selectbox("Inspect DICOM Scans:", dicom_files)
                    fpath = os.path.join(st.session_state.train_dir, sel)
                    dcm = pydicom.dcmread(fpath)
                    pixels = cv2.normalize(dcm.pixel_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    annotated = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)

                    iid = sel.replace('.dicom', '')
                    if df is not None and 'image_id' in df.columns:
                        for _, row in df[df['image_id'] == iid].iterrows():
                            if row.get('class_name') != 'No finding' and not pd.isna(row.get('x_min')):
                                cv2.rectangle(
                                    annotated,
                                    (int(row['x_min']), int(row['y_min'])),
                                    (int(row['x_max']), int(row['y_max'])),
                                    (255, 0, 0), 4
                                )
                    c1, c2 = st.columns(2)
                    c1.image(annotated, caption="Bounding Boxes from Radiologists", use_container_width=True)
                    c2.image(cv2.resize(pixels, (224, 224)), caption="Resized 224x224 AI Input", use_container_width=True)

        elif st.session_state.data_source == "Hugging Face Cloud Stream (IterableDataset)":
            st.subheader("🔍 Hugging Face Dataset Inspector")
            st.info(f"Connected to Repository: `{st.session_state.get('hf_repo', '')}` | Config: `{st.session_state.get('hf_config', 'None')}`")

            num_samples = st.slider("Samples to inspect:", 1, 8, 4)
            if st.button("Fetch & Preview Stream Samples", use_container_width=True):
                with st.spinner("Fetching preview samples (nothing is saved to disk)..."):
                    sample_batch = []
                    try:
                        repo_id = st.session_state.hf_repo
                        config_name = st.session_state.get('hf_config', '').strip()

                        # FIX: try the lightweight REST preview first. It does not touch the
                        # `datasets` library, so it avoids the "must be called with a dataclass
                        # type or instance" error entirely.
                        try:
                            sample_batch = fetch_hf_preview(repo_id, config_name or None, n=num_samples)
                        except Exception as rest_err:
                            st.caption(f"REST preview unavailable ({rest_err}); falling back to `datasets` streaming.")
                            inspect_stream = get_hf_stream(repo_id, config_name)
                            for idx, sample in enumerate(inspect_stream):
                                if idx >= num_samples:
                                    break
                                sample_batch.append(sample)

                        if sample_batch:
                            st.markdown("### 🖼️ Sample Stream Output")
                            st.caption(f"Columns found: {', '.join(sample_batch[0].keys())}")
                            grid_cols = st.columns(len(sample_batch))
                            active_classes = st.session_state.get('active_classes', DEFAULT_DISEASE_CLASSES)

                            for idx, sample in enumerate(sample_batch):
                                img_obj = to_pil_image(sample.get('image', sample.get('img', sample.get('xray', None))))

                                raw_lbl = sample.get('labels', sample.get('label', sample.get('target', 'N/A')))
                                formatted_lbl = str(raw_lbl)
                                if isinstance(raw_lbl, (int, float, np.integer)) and int(raw_lbl) < len(active_classes):
                                    formatted_lbl = f"{raw_lbl} ({active_classes[int(raw_lbl)]})"
                                elif isinstance(raw_lbl, list):
                                    formatted_lbl = ", ".join([active_classes[int(i)] if isinstance(i, (int, float, np.integer)) and int(i) < len(active_classes) else str(i) for i in raw_lbl])

                                grid_cols[idx].image(img_obj, caption=f"Sample #{idx+1}\nLabel: {formatted_lbl}", use_container_width=True)
                            st.warning("Label names above use the default 14-class order. Verify the dataset's own label order before training.")
                    except Exception as e:
                        st.error("Failed to inspect stream. Full traceback below:")
                        st.exception(e)
        else:
            st.info(f"Dataset Inspection for {st.session_state.data_source} is ready. Select 'AI Training Loop' or 'Clinical Evaluation' to execute batch processing.")

# --- TAB 3: AI TRAINING LOOP & VALIDATION ---
if "AI Training Loop" in tab_map:
    with tab_map["AI Training Loop"]:
        st.header("Multi-Label Model Training (4GB VRAM Optimized)")
        col_arch, col_b, col_frz = st.columns(3)
        with col_arch:
            arch_choice = st.selectbox("Backbone Architecture:", ["DenseNet121", "EfficientNet-B0", "MobileNetV3-Large"])
        with col_b:
            target_batches = st.slider("Batches to Train:", 10, 500, 100, step=10)
        with col_frz:
            freeze_base = st.checkbox("Freeze Base Network (Saves VRAM)", value=True)

        if st.button("Train Multi-Label Model"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.info(f"Targeting Device: {device} | Using Automatic Mixed Precision (fp16) | Dataset Engine: {st.session_state.data_source}")

            # --- Dataset & DataLoader Instantiation ---
            with st.spinner(f"Configuring {st.session_state.data_source}..."):

                # MODE 0: Local Disk Datasets
                if st.session_state.data_source == "Local Files":
                    active_classes = st.session_state.get('active_classes', DEFAULT_DISEASE_CLASSES)
                    master_dataset = PyTorchChestXrayDataset(df, st.session_state.train_dir, classes=active_classes, is_train=True)
                    train_size = int(0.8 * len(master_dataset)) if len(master_dataset) > 0 else 0
                    val_size = len(master_dataset) - train_size
                    if train_size > 0:
                        train_indices, val_indices = random_split(range(len(master_dataset)), [train_size, val_size])
                        train_dataset = Subset(master_dataset, train_indices)
                        val_dataset = Subset(PyTorchChestXrayDataset(df, st.session_state.train_dir, classes=active_classes, is_train=False), val_indices)
                    else:
                        train_dataset = master_dataset
                        val_dataset = master_dataset

                    train_loader = TorchDataLoader(train_dataset, batch_size=8, shuffle=True if len(master_dataset) > 0 else False, num_workers=0)
                    val_loader = TorchDataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

                # MODE 1: Hugging Face IterableDataset Streaming
                elif st.session_state.data_source == "Hugging Face Cloud Stream (IterableDataset)":
                    try:
                        hf_stream = get_hf_stream(st.session_state.hf_repo, st.session_state.get('hf_config', ''))
                        active_classes = st.session_state.get('active_classes', DEFAULT_DISEASE_CLASSES)

                        # FIX: no more train/validation leakage. Validation uses the first
                        # HF_VAL_SIZE rows, training uses everything after them.
                        hf_val_stream = hf_stream.take(HF_VAL_SIZE)
                        hf_train_stream = hf_stream.skip(HF_VAL_SIZE).shuffle(seed=42, buffer_size=500)

                        train_dataset = HFStreamingChestXrayDataset(hf_train_stream, classes=active_classes, is_train=True)
                        val_dataset = HFStreamingChestXrayDataset(hf_val_stream, classes=active_classes, is_train=False)

                        train_loader = TorchDataLoader(train_dataset, batch_size=8)
                        val_loader = TorchDataLoader(val_dataset, batch_size=8)
                    except Exception as e:
                        st.error("HF Streaming Error. Full traceback below:")
                        st.exception(e)
                        st.stop()

                # MODE 2: WebDataset (.tar Shards)
                elif st.session_state.data_source == "WebDataset (.tar HTTP/S3 Shards)":
                    if not WEBDATASET_AVAILABLE:
                        st.error("Please install `webdataset`.")
                        st.stop()
                    try:
                        active_classes = st.session_state.get('active_classes', DEFAULT_DISEASE_CLASSES)
                        train_dataset = create_webdataset_pipeline(st.session_state.wds_url, classes=active_classes, is_train=True)
                        val_dataset = create_webdataset_pipeline(st.session_state.wds_url, classes=active_classes, is_train=False)

                        train_loader = TorchDataLoader(train_dataset, batch_size=8)
                        val_loader = TorchDataLoader(val_dataset, batch_size=8)
                    except Exception as e:
                        st.error("WebDataset Error. Full traceback below:")
                        st.exception(e)
                        st.stop()

                # MODE 3: MosaicML Streaming Dataset
                elif st.session_state.data_source == "MosaicML Streaming (MDS Format)":
                    if not MOSAIC_AVAILABLE:
                        st.error("Please install `mosaicml-streaming`.")
                        st.stop()
                    try:
                        active_classes = st.session_state.get('active_classes', DEFAULT_DISEASE_CLASSES)
                        train_dataset = MosaicStreamingChestXray(
                            remote_url=st.session_state.mosaic_remote,
                            local_cache=st.session_state.mosaic_cache,
                            classes=active_classes,
                            is_train=True
                        )
                        val_dataset = MosaicStreamingChestXray(
                            remote_url=st.session_state.mosaic_remote,
                            local_cache=st.session_state.mosaic_cache,
                            classes=active_classes,
                            is_train=False
                        )

                        train_loader = TorchDataLoader(train_dataset, batch_size=8, num_workers=2)
                        val_loader = TorchDataLoader(val_dataset, batch_size=8, num_workers=2)
                    except Exception as e:
                        st.error("MosaicML Streaming Error. Full traceback below:")
                        st.exception(e)
                        st.stop()

            # Dynamic Model Classifier Head Initialization
            active_classes = st.session_state.get('active_classes', DEFAULT_DISEASE_CLASSES)
            model, default_layer = build_model(
                model_name=arch_choice,
                num_classes=len(active_classes),
                pretrained=True,
                freeze_backbone=freeze_base
            )

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if freeze_base else 1e-4)
            scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

            val_iter = iter(val_loader)
            prog = st.progress(0)
            status = st.empty()
            chart = st.empty()
            history_data = {"Train Loss": [], "Validation Loss": []}

            for i, (images, labels) in enumerate(train_loader):
                if i >= target_batches:
                    break

                model.train()
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                model.eval()
                try:
                    v_imgs, v_lbls = next(val_iter)
                except (StopIteration, Exception):
                    val_iter = iter(val_loader)
                    try:
                        v_imgs, v_lbls = next(val_iter)
                    except Exception:
                        v_imgs, v_lbls = images, labels

                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                        v_out = model(v_imgs.to(device, non_blocking=True))
                        v_loss = criterion(v_out, v_lbls.to(device, non_blocking=True))

                history_data["Train Loss"].append(loss.item())
                history_data["Validation Loss"].append(v_loss.item())

                prog.progress(int(((i + 1) / target_batches) * 100))
                status.text(f"Batch {i+1}/{target_batches} | Train Loss: {loss.item():.4f} | Val Loss: {v_loss.item():.4f}")
                chart.line_chart(pd.DataFrame(history_data))

            torch.save({
                'arch': arch_choice,
                'num_classes': len(active_classes),
                'state_dict': model.state_dict()
            }, "shared_weights.pth")
            load_shared_model.clear()
            st.success(f"Successfully trained & saved {arch_choice} master weights!")

        if os.path.exists("shared_weights.pth"):
            st.markdown("---")
            with open("shared_weights.pth", "rb") as f:
                st.download_button(
                    label="⬇️ Download Trained Model Weights (.pth)",
                    data=f,
                    file_name="medical_ai_multi_label_weights.pth",
                    mime="application/octet-stream"
                )

# --- TAB 4: EXPLAINABILITY (MULTI-LABEL GRAD-CAM) ---
if "AI Diagnostics" in tab_map:
    with tab_map["AI Diagnostics"]:
        st.header("Grad-CAM: Disease-Specific Spatial Attention")
        shared_model, active_arch, def_layer = load_shared_model()
        active_classes = st.session_state.get('active_classes', DEFAULT_DISEASE_CLASSES)

        if shared_model is None:
            st.warning("Please train a model in Tab 3 or upload weights first.")
        else:
            class MultiLayerGradCAM:
                def __init__(self, model):
                    self.model = model
                    self.gradients = None
                    self.activations = None

                def get_hook(self):
                    def hook(module, input, output):
                        self.activations = output.detach().clone()
                    return hook

                def get_backward_hook(self):
                    def hook(module, grad_in, grad_out):
                        self.gradients = grad_out[0].detach().clone()
                    return hook

                def generate_heatmap(self, input_tensor, layer_name, class_idx=None):
                    for p in self.model.parameters():
                        p.requires_grad = True
                    input_tensor.requires_grad = True

                    target_layer = None
                    for name, module in self.model.features.named_modules():
                        if name == layer_name or name.endswith(layer_name):
                            target_layer = module
                            break
                    if target_layer is None:
                        target_layer = list(self.model.features.children())[-1]

                    fh = target_layer.register_forward_hook(self.get_hook())
                    bh = target_layer.register_full_backward_hook(self.get_backward_hook())

                    self.model.eval()
                    output = self.model(input_tensor)
                    probabilities = torch.sigmoid(output)
                    self.model.zero_grad()

                    if class_idx is None:
                        class_idx = torch.argmax(probabilities, dim=1).item()

                    target_score = probabilities[0, class_idx]
                    target_score.backward()

                    if self.gradients is None:
                        fh.remove(); bh.remove()
                        return np.zeros((224, 224)), probabilities.detach().cpu().numpy()[0], class_idx

                    acts = self.activations.cpu().numpy()[0]
                    grads = self.gradients.cpu().numpy()[0]
                    weights = np.mean(grads, axis=(1, 2))

                    cam = np.zeros(acts.shape[1:], dtype=np.float32)
                    for i, w in enumerate(weights):
                        cam += w * acts[i]
                    cam = np.maximum(cam, 0)
                    cam = cv2.resize(cam, (224, 224))
                    if np.max(cam) != 0:
                        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

                    fh.remove(); bh.remove()
                    return cam, probabilities.detach().cpu().numpy()[0], class_idx

            col1, col2 = st.columns([1, 2])
            with col1:
                uploaded = st.file_uploader("Upload Chest Scan (DICOM/PNG)", type=['dicom', 'png', 'jpg'])
                target_options = ["Auto (Highest Probability)"] + active_classes
                target_disease = st.selectbox("Inspect Localization For:", target_options)
                opacity = st.slider("Heatmap Opacity:", 0.1, 1.0, 0.4, step=0.1)

            with col2:
                if uploaded and st.button("Generate Diagnostic Report", use_container_width=True):
                    fbytes = uploaded.read()

                    with st.spinner("Analyzing scan and generating diagnostic heatmaps..."):
                        if uploaded.name.endswith('.dicom') and PYDICOM_AVAILABLE:
                            dcm = pydicom.dcmread(io.BytesIO(fbytes))
                            norm = cv2.normalize(dcm.pixel_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
                        else:
                            rgb = np.array(Image.open(io.BytesIO(fbytes)).convert('RGB'))

                        transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        tensor = transform(rgb).unsqueeze(0).to(device)

                        gradcam = MultiLayerGradCAM(shared_model)

                        c_idx = None if target_disease == "Auto (Highest Probability)" else active_classes.index(target_disease)
                        heatmap, predictions, best_idx = gradcam.generate_heatmap(tensor, def_layer, c_idx)
                        detected_disease = active_classes[best_idx]

                        st.markdown("### 📋 AI Diagnostic Breakdown")
                        ranked = sorted(zip(active_classes, predictions), key=lambda x: x[1], reverse=True)
                        t_cols = st.columns(min(3, len(active_classes)))
                        for idx, (d_name, score) in enumerate(ranked[:6]):
                            t_cols[idx % len(t_cols)].metric(d_name, f"{score * 100:.2f}%", delta="Abnormal" if score > 0.4 else "Normal")

                        resized_orig = cv2.resize(rgb, (224, 224))
                        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(resized_orig, 1.0 - opacity, heatmap_color, opacity, 0)

                        c_a, c_b, c_c = st.columns(3)
                        c_a.image(resized_orig, caption="Original X-Ray", use_container_width=True)
                        c_b.image(heatmap_color, caption=f"Heatmap: {detected_disease}", use_container_width=True)
                        c_c.image(overlay, caption="Superimposed Focus", use_container_width=True)

# --- TAB 5: CLINICAL EVALUATION ---
if "Clinical Evaluation" in tab_map:
    with tab_map["Clinical Evaluation"]:
        st.header("Multi-Disease Validation & Evaluation")
        shared_model, active_arch, _ = load_shared_model()
        active_classes = st.session_state.get('active_classes', DEFAULT_DISEASE_CLASSES)

        if shared_model is None:
            st.warning("Please configure your dataset and train/upload a model first.")
        else:
            if st.button("Run Multi-Label Evaluation"):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                shared_model.to(device)
                shared_model.eval()
                eval_loader = None

                with st.spinner(f"Preparing validation stream via {st.session_state.data_source}..."):
                    try:
                        if st.session_state.data_source == "Local Files":
                            val_dataset = PyTorchChestXrayDataset(df, st.session_state.train_dir, classes=active_classes, is_train=False)
                            eval_size = min(150, len(val_dataset)) if len(val_dataset) > 0 else 0
                            if eval_size > 0:
                                eval_subset, _ = random_split(val_dataset, [eval_size, len(val_dataset) - eval_size])
                            else:
                                eval_subset = val_dataset
                            eval_loader = TorchDataLoader(eval_subset, batch_size=8, shuffle=False)
                        elif st.session_state.data_source == "Hugging Face Cloud Stream (IterableDataset)":
                            hf_stream = get_hf_stream(st.session_state.hf_repo, st.session_state.get('hf_config', ''))
                            # Same held-out rows used as validation during training
                            hf_val_stream = hf_stream.take(HF_VAL_SIZE)
                            eval_loader = TorchDataLoader(HFStreamingChestXrayDataset(hf_val_stream, classes=active_classes, is_train=False), batch_size=8)
                        elif st.session_state.data_source == "WebDataset (.tar HTTP/S3 Shards)":
                            eval_loader = TorchDataLoader(create_webdataset_pipeline(st.session_state.wds_url, classes=active_classes, is_train=False), batch_size=8)
                        elif st.session_state.data_source == "MosaicML Streaming (MDS Format)":
                            eval_loader = TorchDataLoader(MosaicStreamingChestXray(st.session_state.mosaic_remote, st.session_state.mosaic_cache, classes=active_classes, is_train=False), batch_size=8)
                    except Exception as e:
                        st.error("Could not prepare the evaluation stream. Full traceback below:")
                        st.exception(e)

                all_preds, all_labels = [], []

                if eval_loader is not None:
                    with st.spinner("Running multi-label validation inference..."):
                        with torch.no_grad():
                            for b_idx, (imgs, lbls) in enumerate(eval_loader):
                                if b_idx >= 20:
                                    break
                                imgs = imgs.to(device, non_blocking=True)
                                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                                    out = shared_model(imgs)

                                probs = torch.sigmoid(out.float())
                                all_preds.append(probs.cpu().numpy())
                                all_labels.append(lbls.numpy())

                if len(all_preds) > 0:
                    all_preds = np.vstack(all_preds)
                    all_labels = np.vstack(all_labels)

                    st.markdown("### 📊 ROC Curves (Area Under Curve by Class)")
                    fig, ax = plt.subplots(figsize=(10, 7))
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

                    auc_records = []
                    for i, c_name in enumerate(active_classes):
                        if len(np.unique(all_labels[:, i])) > 1:
                            fpr, tpr, _ = roc_curve(all_labels[:, i], all_preds[:, i])
                            roc_auc = auc(fpr, tpr)
                            auc_records.append({"Disease": c_name, "AUC": round(roc_auc, 3)})
                            ax.plot(fpr, tpr, lw=2, label=f'{c_name} (AUC = {roc_auc:.2f})')

                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
                    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
                    ax.set_title('Medical AI Diagnostic Performance (Multi-Class)', fontsize=14)

                    if len(auc_records) > 0:
                        ax.legend(loc="lower right", fontsize=9, bbox_to_anchor=(1.35, 0.0))
                        ax.grid(alpha=0.3)
                        st.pyplot(fig)
                        st.markdown("---")
                        st.dataframe(pd.DataFrame(auc_records), use_container_width=True)
                    else:
                        st.info("Evaluation stream contained single-class presence. Stream more batches to plot ROC.")
