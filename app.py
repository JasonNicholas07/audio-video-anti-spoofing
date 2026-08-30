import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import librosa
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_PATH = "blaze_face_short_range.tflite"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def _ensure_face_model():
    if not os.path.exists(_MODEL_PATH):
        import urllib.request
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


@st.cache_resource
def load_face_detector():
    model_path = _ensure_face_model()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options, min_detection_confidence=0.5
    )
    return mp_vision.FaceDetector.create_from_options(options)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
N_FRAMES = 15  
VIDEO_CKPT = "model/video/best_cbam_gru_augmented.pt"
AUDIO_CKPT = "model/audio/best_audio_model.pt"
AUDIO_SAMPLE_RATE = 16000
AUDIO_CLIP_DURATION = 3.5  
AUDIO_N_MELS = 128
AUDIO_CLIP_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_CLIP_DURATION)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False), nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class CBAM_GRU_DeepfakeDetector(nn.Module):
    def __init__(self, backbone_checkpoint=None, feature_dim=1280, gru_hidden=256, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=(backbone_checkpoint is None),
            num_classes=0, global_pool="",
        )
        self.cbam = CBAM(channels=feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gru = nn.GRU(feature_dim, gru_hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(gru_hidden * 2, num_classes)

    def forward(self, x, lengths):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat_map = self.backbone.forward_features(x)
        feat_map = self.cbam(feat_map)
        feat_vec = self.pool(feat_map).flatten(1).view(B, T, -1)
        packed = nn.utils.rnn.pack_padded_sequence(feat_vec, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(h_cat)


VAL_TRANSFORM = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])



class AudioCNN(nn.Module):
    """Matches audio_train.py exactly — mel-spectrogram CNN."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.classifier(x)

@st.cache_resource
def load_video_model():
    model = CBAM_GRU_DeepfakeDetector()
    if os.path.exists(VIDEO_CKPT):
        state = torch.load(VIDEO_CKPT, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        st.warning(f"Video checkpoint not found at {VIDEO_CKPT} — using untrained weights.")
    model.to(DEVICE).eval()
    return model


@st.cache_resource
def load_audio_model():
    model = AudioCNN()
    if os.path.exists(AUDIO_CKPT):
        state = torch.load(AUDIO_CKPT, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        st.warning(f"Audio checkpoint not found at {AUDIO_CKPT} — using untrained weights.")
    model.to(DEVICE).eval()
    return model


def crop_face(frame_rgb, face_detector, margin=0.2, output_size=IMG_SIZE, center_fallback=True):
    """Detect + crop the largest face. If center_fallback is True, falls back
    to a center crop when no face is found; if False, returns None (matches
    live_test_both_modal.py's detect_and_crop_face, which returns None)."""
    h, w, _ = frame_rgb.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = face_detector.detect(mp_image)

    if result.detections:
        det = max(result.detections, key=lambda d: d.categories[0].score)
        bb = det.bounding_box
        x1 = max(int(bb.origin_x - margin * bb.width), 0)
        y1 = max(int(bb.origin_y - margin * bb.height), 0)
        x2 = min(int(bb.origin_x + bb.width * (1 + margin)), w)
        y2 = min(int(bb.origin_y + bb.height * (1 + margin)), h)
        if x2 > x1 and y2 > y1:
            crop = frame_rgb[y1:y2, x1:x2]
            return cv2.resize(crop, (output_size, output_size))

    if not center_fallback:
        return None

    # fallback: center crop
    side = min(h, w)
    y1, x1 = (h - side) // 2, (w - side) // 2
    crop = frame_rgb[y1:y1 + side, x1:x1 + side]
    return cv2.resize(crop, (output_size, output_size))


def extract_frames(video_path, face_detector, n_frames=N_FRAMES, start_offset_ratio=0.1):
    """Matches live_test_both_modal.py's extract_consecutive_crops —
    consecutive frames from a start offset, carrying forward the last valid
    crop when a frame's face detection fails (not evenly-spaced sampling)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    start_frame = int(total * start_offset_ratio)
    start_frame = min(start_frame, max(0, total - n_frames))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    crops = []
    last_valid = None
    saved = 0
    while saved < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cropped = crop_face(frame_rgb, face_detector, center_fallback=False)
        if cropped is not None:
            last_valid = cropped
            crops.append(cropped)
            saved += 1
        elif last_valid is not None:
            crops.append(last_valid)
            saved += 1

    cap.release()
    return crops


FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"  

def check_ffmpeg_installed():
    from shutil import which
    return FFMPEG_PATH != "ffmpeg" or which("ffmpeg") is not None


def extract_audio_from_video(video_path, output_wav_path, sample_rate=AUDIO_SAMPLE_RATE):
    import subprocess
    cmd = [
        FFMPEG_PATH, "-y", "-i", video_path,
        "-vn", "-ar", str(sample_rate), "-ac", "1",
        output_wav_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return False, "ffmpeg is not installed or not on PATH."
    if result.returncode != 0:
        return False, result.stderr.strip()[-500:]
    return True, None


def fix_clip_length(y, target_samples=AUDIO_CLIP_SAMPLES):
    """Matches live_test_both_modal.py's load_audio_fixed_length —
    center crop for inference (training used random crop, inference doesn't)."""
    if len(y) > target_samples:
        start = (len(y) - target_samples) // 2
        y = y[start:start + target_samples]
    else:
        pad = target_samples - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    return y


def audio_to_melspec(y, sr=AUDIO_SAMPLE_RATE, n_mels=AUDIO_N_MELS):
    """Matches audio_train.py's audio_to_melspec exactly."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)


def predict_video(model, frames):
    if not frames:
        return None
    tensors = [VAL_TRANSFORM(image=f)["image"] for f in frames]
    seq = torch.stack(tensors).unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)
    lengths = torch.tensor([seq.shape[1]])
    with torch.no_grad():
        logits = model(seq, lengths)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs  # [P(real), P(fake)]


def predict_audio(model, waveform):
    if waveform is None:
        return None
    y = fix_clip_length(waveform)
    mel = audio_to_melspec(y)
    mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, n_mels, T)
    with torch.no_grad():
        logits = model(mel_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs  # [P(real), P(fake)]


def combine_modalities(video_prob, audio_prob, video_weight=0.6, audio_weight=0.4,
                        extreme_high=0.85, extreme_low=0.15):
    """Matches live_test_both_modal.py's combine_modalities exactly."""
    if video_prob is None and audio_prob is None:
        return None, "no_signal"
    if video_prob is None:
        return audio_prob, "audio_only"
    if audio_prob is None:
        return video_prob, "video_only"

    video_extreme = video_prob >= extreme_high or video_prob <= extreme_low
    audio_extreme = audio_prob >= extreme_high or audio_prob <= extreme_low

    if video_extreme or audio_extreme:
        video_distance = abs(video_prob - 0.5)
        audio_distance = abs(audio_prob - 0.5)
        if video_distance >= audio_distance:
            return video_prob, "video_dominant (extreme)"
        else:
            return audio_prob, "audio_dominant (extreme)"
    else:
        combined = video_weight * video_prob + audio_weight * audio_prob
        return combined, "weighted_average"


VIDEO_THRESHOLD = 0.6  

st.set_page_config(page_title="Pendeteksi Deepfake Audio Video")
st.title("Pendeteksi Deepfake Audio Video")
st.caption("Tes Video anda")

if not check_ffmpeg_installed():
    st.warning(
        "ffmpeg was not found"
    )

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("Run detection"):
        video_model = load_video_model()
        audio_model = load_audio_model()
        face_detector = load_face_detector()

        with st.spinner("Extracting frames and analyzing video..."):
            frames = extract_frames(video_path, face_detector)
            video_probs = predict_video(video_model, frames)

        audio_error = None
        audio_probs = None
        with st.spinner("Extracting audio and analyzing..."):
            wav_path = video_path + "_audio.wav"
            ok, err = extract_audio_from_video(video_path, wav_path)
            if ok and os.path.exists(wav_path):
                waveform, _ = librosa.load(wav_path, sr=AUDIO_SAMPLE_RATE)
                os.remove(wav_path)
                audio_probs = predict_audio(audio_model, waveform)
            else:
                audio_error = err or "Audio extraction failed for an unknown reason."

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Video analysis")
            if video_probs is not None:
                fake_pct = video_probs[1] * 100
                st.metric("Fake probability", f"{fake_pct:.1f}%")
                st.progress(min(int(fake_pct), 100))
                verdict = "Kemungkinan dimanipulasi" if video_probs[1] >= VIDEO_THRESHOLD else " Kemungkinan asli"
                st.write(verdict if verdict == "Kemungkinan dimanipulasi" else verdict)
            else:
                st.error("Tidak bisa deteksi wajah")

        with col2:
            st.subheader("Audio analysis")
            if audio_probs is not None:
                fake_pct = audio_probs[1] * 100
                st.metric("Fake probability", f"{fake_pct:.1f}%")
                st.progress(min(int(fake_pct), 100))
                st.write("Kemungkinan  dimanipulasi" if fake_pct > 50 else "Kemungkinan asli")
            else:
                st.error(f"Tidak bisa extract audio: {audio_error}")

        st.divider()
        video_p = video_probs[1] if video_probs is not None else None
        audio_p = audio_probs[1] if audio_probs is not None else None
        combined, method = combine_modalities(video_p, audio_p)

        st.subheader("Overall verdict")
        if combined is not None:
            st.caption(f"Combination method: **{method}**")
            verdict = "Kemungkinan palsu" if combined >= 0.5 else "Kemungkinan asli"
            if verdict == "Kemungkinan palsu":
                st.error(f"{verdict} ({combined*100:.1f}% confidence)")
            else:
                st.success(f"{verdict} ({combined*100:.1f}% confidence)")
        else:
            st.info("No usable video")

    os.remove(video_path)
else:
    st.info("Upload MP4/MOV/AVI/MKV file untuk tes")