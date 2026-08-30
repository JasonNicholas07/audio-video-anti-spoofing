import streamlit as st
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import cv2
import librosa
import requests
import subprocess
import albumentations as A
from albumentations.pytorch import ToTensorV2

# CONFIG
DATA_FILE = "survey_results.csv"

VIDEO_CHECKPOINT = "model/video/best_cbam_gru_augmented.pt"
AUDIO_CHECKPOINT = "model/audio/best_audio_model.pt"
FACE_MODEL_PATH = "model/blaze_face_short_range.tflite"
BEST_VAL_THRESHOLD = 0.5 

SET_A = [
    ("SMS: 'BCA: Rekening Anda akan ditangguhkan dalam 24 jam. Verifikasi sekarang: bit.ly/bca-verify88'", True),
    ("Email dari bank Anda yang meminta Anda untuk masuk melalui aplikasi resmi guna memeriksa laporan mutasi baru.", False),
    ("Pesan WhatsApp dari nomor tak dikenal: 'Bu, HP ku hilang, ini nomor baruku, tolong transfer Rp2.000.000 segera.'", True),
    ("Panggilan dari nomor yang mengaku sebagai bagian penanganan penipuan bank Anda, yang meminta Anda menyebutkan kembali kode OTP yang baru saja Anda terima.", True),
    ("Notifikasi di dalam aplikasi perbankan Anda mengenai transfer berhasil yang baru saja Anda lakukan sendiri.", False),
    ("SMS yang menyatakan Anda memenangkan undian yang tidak pernah Anda ikuti, serta meminta biaya administrasi untuk mengklaim hadiah tersebut.", True),
    ("Email dari alamat email asli rekan kerja, membahas proyek yang sedang kalian kerjakan bersama.", False),
    ("Kode QR di kasir toko yang mengarahkan ke halaman pembayaran umum, bukan ke layar konfirmasi QRIS resmi bank Anda.", True),
]

SET_B = [
    ("SMS: 'Paket Anda tidak dapat dikirim. Bayar biaya bea cukai sebesar Rp15.000 di sini: shorturl.at/pkg2024'", True),
    ("Notifikasi push dari sistem operasi ponsel yang mengingatkan bahwa daya baterai Anda lemah.", False),
    ("Panggilan video dari seseorang yang wajah dan suaranya mirip atasan Anda, yang dengan mendesak meminta Anda mentransfer uang ke rekening vendor baru sebelum hari berakhir.", True),
    ("Pesan teks dari teman yang mengonfirmasi rencana makan malam yang telah kalian bicarakan kemarin.", False),
    ("Email yang mengatasnamakan perusahaan jasa pengiriman, meminta Anda untuk 'mengonfirmasi ulang detail pembayaran' melalui sebuah tautan.", True),
    ("Panggilan dari nomor resmi bank Anda yang mengonfirmasi transaksi yang Anda lakukan, tanpa permintaan tindakan apa pun.", False),
    ("Pesan yang mengatasnamakan dukungan teknis, menyatakan bahwa komputer Anda terkena virus dan meminta akses jarak jauh.", True),
    ("Kode OTP via SMS tanpa permintaan tambahan, hanya kodenya saja yang dikirim setelah Anda mencoba masuk (login).", False),
]

CONFIDENCE_QUESTIONS = [
    "Seberapa yakin Anda dalam kemampuan Anda untuk mengenali pesan atau panggilan penipuan?",
    "Seberapa yakin Anda dalam memverifikasi apakah kode QR atau tautan pembayaran itu sah?",
]

# Pre-test: user judges ALONE, no AI help
DEEPFAKE_SET_A = [
    # (path, media_type: "video"/"audio", is_fake)
    ("live_test/questionnaire/fake001.mp4", "video", True),
    ("live_test/questionnaire/real_audio6.wav", "audio", False),
    ("live_test/questionnaire/fake001.wav", "audio", True),
    ("live_test/questionnaire/real001.mp4", "video", False),
    ("live_test/questionnaire/real001.wav", "audio", False),
    ("live_test/questionnaire/fake_video7.mp4", "video", True),
]

# Post-test: user judges WITH the AI model's assistance shown per clip
DEEPFAKE_SET_B = [
    ("live_test/questionnaire/real6.mp4", "video", False),    
    ("live_test/questionnaire/real7.mp4", "video", False),
    ("live_test/questionnaire/fake3.mp4", "video", True),
    ("live_test/questionnaire/real003.mp4", "video", False),
    ("live_test/questionnaire/real5.mp4", "video", False),
    ("live_test/questionnaire/fake.mp4", "video", True),   
]

INTENT_QUESTIONS = [
    "Apakah Anda akan memverifikasi pesan yang mencurigakan sebelum menindaklanjutinya, meskipun tampaknya mendesak?",
    "Apakah Anda akan menggunakan alat seperti ini (deteksi penipuan berbasis AI) jika terintegrasi ke dalam aplikasi perbankan Anda?",
]


# MODEL DEFINITIONS
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
    def __init__(self, feature_dim=1280, gru_hidden=256, num_classes=2):
        super().__init__()
        import timm
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False,
                                           num_classes=0, global_pool="")
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
        packed = nn.utils.rnn.pack_padded_sequence(
            feat_vec, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(h_cat)


class AudioCNN(nn.Module):
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
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_model = CBAM_GRU_DeepfakeDetector().to(device)
    audio_model = None

    if os.path.exists(VIDEO_CHECKPOINT):
        video_model.load_state_dict(torch.load(VIDEO_CHECKPOINT, map_location=device))
        video_model.eval()
    else:
        video_model = None

    if os.path.exists(AUDIO_CHECKPOINT):
        audio_model = AudioCNN().to(device)
        audio_model.load_state_dict(torch.load(AUDIO_CHECKPOINT, map_location=device))
        audio_model.eval()

    return video_model, audio_model, device


@st.cache_resource
def load_face_detector():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not os.path.exists(FACE_MODEL_PATH):
        subprocess.run([
            "wget", "-q", "-O", FACE_MODEL_PATH,
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        ])
    base_options = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
    return vision.FaceDetector.create_from_options(options)


def get_video_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def extract_consecutive_crops(video_path, detector, n_consecutive=15, start_offset_ratio=0.1):
    import mediapipe as mp
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    start_frame = int(total_frames * start_offset_ratio)
    start_frame = min(start_frame, max(0, total_frames - n_consecutive))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    crops, last_valid, saved = [], None, 0
    while saved < n_consecutive:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        cropped = None
        if result.detections:
            best = max(result.detections, key=lambda d: d.categories[0].score)
            bbox = best.bounding_box
            margin = 0.2
            x1 = max(0, int(bbox.origin_x - bbox.width * margin))
            y1 = max(0, int(bbox.origin_y - bbox.height * margin))
            x2 = min(w, int(bbox.origin_x + bbox.width * (1 + margin)))
            y2 = min(h, int(bbox.origin_y + bbox.height * (1 + margin)))
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                cropped = cv2.resize(face, (224, 224))
        if cropped is not None:
            last_valid = cropped
            crops.append(cropped)
            saved += 1
        elif last_valid is not None:
            crops.append(last_valid)
            saved += 1
    cap.release()
    return crops


def predict_video(video_path, video_model, detector, transform, device):
    crops = extract_consecutive_crops(video_path, detector)
    if len(crops) == 0:
        return None
    frames = [transform(image=cv2.cvtColor(c, cv2.COLOR_BGR2RGB))["image"] for c in crops]
    seq = torch.stack(frames).unsqueeze(0).to(device)
    length = torch.tensor([len(frames)])
    with torch.no_grad():
        outputs = video_model(seq, length)
        prob = torch.softmax(outputs, dim=1)[0, 1].item()
    return prob


def predict_audio(audio_path, audio_model, device, sr=16000, duration=3.5, n_mels=128):
    y, _ = librosa.load(audio_path, sr=sr)
    target_samples = int(sr * duration)
    if len(y) > target_samples:
        start = (len(y) - target_samples) // 2
        y = y[start:start + target_samples]
    else:
        y = np.pad(y, (0, target_samples - len(y)), mode="constant")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_tensor = torch.tensor(mel_db.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = audio_model(mel_tensor)
        prob = torch.softmax(outputs, dim=1)[0, 1].item()
    return prob


def get_ai_verdict(path, media_type):
    video_model, audio_model, device = load_models()
    if media_type == "video" and video_model is not None:
        detector = load_face_detector()
        transform = get_video_transform()
        prob = predict_video(path, video_model, detector, transform, device)
    elif media_type == "audio" and audio_model is not None:
        prob = predict_audio(path, audio_model, device)
    else:
        prob = None

    if prob is None:
        return None, "Model tidak tersedia atau gagal memproses file ini."
    verdict = "PALSU (AI-generated)" if prob >= BEST_VAL_THRESHOLD else "ASLI"
    return prob, verdict


# HELPERS — survey logic
def _airtable_headers():
    return {
        "Authorization": f"Bearer {st.secrets['airtable']['token']}",
        "Content-Type": "application/json",
    }

def _airtable_url():
    return f"https://api.airtable.com/v0/{st.secrets['airtable']['base_id']}/{st.secrets['airtable']['table_name']}"

def init_data_file():
    pass


def save_result(row: dict):
    clean_row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    requests.post(_airtable_url(), headers=_airtable_headers(), json={"fields": clean_row})


def render_detection_quiz(question_set, key_prefix):
    st.write("Untuk setiap item, tentukan apakah ini penipuan atau sah/asli?")
    answers = {}
    for i, (text, _) in enumerate(question_set):
        st.markdown(f"**{i+1}.** {text}")
        answers[i] = st.radio(
            "Your answer:", ["Palsu", "Asli", "Tidak yakin"],
            key=f"{key_prefix}_q{i}", horizontal=True, label_visibility="collapsed"
        )
        st.write("")
    return answers


def score_detection(answers, question_set):
    correct = 0
    for i, (_, is_scam) in enumerate(question_set):
        given = answers.get(i)
        correct_label = "Palsu" if is_scam else "Asli"
        if given == correct_label:
            correct += 1
    return correct, len(question_set)


def render_likert_block(questions, key_prefix):
    scores = []
    for i, q in enumerate(questions):
        val = st.slider(q, 1, 5, 3, key=f"{key_prefix}_{i}",
                         help="1 = Sangat tidak setuju, 5 = Sangat Setuju")
        scores.append(val)
    return scores


def render_deepfake_quiz_unassisted(samples, key_prefix):
    """Pre-test: user judges alone, no AI shown."""
    st.write("Dengarkan atau lihatlah setiap klip. Apakah ini asli atau palsu (AI-generated)?")
    answers = {}
    for i, (path, media_type, _) in enumerate(samples):
        st.markdown(f"**Klip {i+1}**")
        if os.path.exists(path):
            if media_type == "video":
                st.video(path)
            else:
                st.audio(path)
        else:
            st.warning(f"File not found: {path}")
        answers[i] = st.radio(
            "Jawaban Anda:", ["Asli", "Palsu", "Tidak yakin"],
            key=f"{key_prefix}_df{i}", horizontal=True, label_visibility="collapsed"
        )
        st.write("")
    return answers


def render_deepfake_quiz_assisted(samples, key_prefix):
    st.write(
        "Dengarkan atau lihatlah setiap klip. Kali ini, sistem AI akan "
        "menampilkan analisisnya, gunakan untuk membantu keputusan akhir Anda."
    )
    answers = {}
    for i, (path, media_type, _) in enumerate(samples):
        st.markdown(f"**Klip {i+1}**")
        if os.path.exists(path):
            if media_type == "video":
                st.video(path)
            else:
                st.audio(path)
        else:
            st.warning(f"File not found: {path}")

        cache_key = f"{key_prefix}_verdict_{i}"
        if cache_key not in st.session_state:
            with st.spinner("AI sedang menganalisis..."):
                prob, verdict = get_ai_verdict(path, media_type)
            st.session_state[cache_key] = (prob, verdict)
        prob, verdict = st.session_state[cache_key]

        if prob is not None:
            st.info(f"Hasil Analisis AI: **{verdict}** (skor kepercayaan palsu: {prob:.1%})")
        else:
            st.warning(verdict)

        answers[i] = st.radio(
            "Keputusan akhir Anda:", ["Asli", "Palsu", "Tidak yakin"],
            key=f"{key_prefix}_df{i}", horizontal=True, label_visibility="collapsed"
        )
        st.write("")
    return answers

def load_all_results():
    records, offset = [], None
    while True:
        params = {"offset": offset} if offset else {}
        resp = requests.get(_airtable_url(), headers=_airtable_headers(), params=params).json()
        records.extend([r["fields"] for r in resp.get("records", [])])
        offset = resp.get("offset")
        if not offset:
            break
    return pd.DataFrame(records)

def score_deepfake_quiz(answers, samples):
    correct = 0
    for i, (_, _, is_fake) in enumerate(samples):
        given = answers.get(i)
        correct_label = "Palsu" if is_fake else "Asli"
        if given == correct_label:
            correct += 1
    return correct, len(samples)


# APP
st.set_page_config(page_title="Kuesioner", layout="centered")
st.title("Kuesioner Pemahaman Penipuan Video / Audio")
st.caption("Pre-test → Demo → Post-test → Hasil")

if "stage" not in st.session_state:
    st.session_state.stage = "intro"
if "participant_id" not in st.session_state:
    st.session_state.participant_id = ""

# --- INTRO ---
if st.session_state.stage == "intro":
    st.write(
        "Studi singkat ini mengukur kemampuan mendeteksi penipuan serta tingkat "
        "keyakinan sebelum dan sesudah melihat demo konsep AI tools. Total waktunya "
        "sekitar 5–7 menit."
    )
    pid = st.text_input("Nama:")
    prior = st.radio(
        "Apakah Anda atau seseorang yang Anda kenal pernah menjadi sasaran penipuan "
        "dalam satu tahun terakhir?",
        ["Iya", "Tidak", "Tidak yakin"]
    )
    if st.button("Mulai Pre-Test", disabled=(pid.strip() == "")):
        st.session_state.participant_id = pid.strip()
        st.session_state.prior_scam_exposure = prior
        st.session_state.stage = "pre_test"
        st.rerun()

# --- PRE-TEST (unassisted) ---
elif st.session_state.stage == "pre_test":
    st.header("Pre-Test")
    st.caption("Jawablah sendiri, tanpa bantuan AI, untuk mengukur kemampuan awal Anda.")
    st.subheader("Part 1 — Deteksi Penipuan")
    pre_answers = render_detection_quiz(SET_A, "pre")

    st.subheader("Part 2 — Video / Audio (tanpa bantuan AI)")
    pre_df_answers = render_deepfake_quiz_unassisted(DEEPFAKE_SET_A, "pre_df")

    st.subheader("Part 3 — Tingkat Keyakinan")
    pre_confidence = render_likert_block(CONFIDENCE_QUESTIONS, "pre_conf")

    if st.button("Submit Pre-Test"):
        score, total = score_detection(pre_answers, SET_A)
        df_score, df_total = score_deepfake_quiz(pre_df_answers, DEEPFAKE_SET_A)
        st.session_state.pre_score = score
        st.session_state.pre_total = total
        st.session_state.pre_df_score = df_score
        st.session_state.pre_df_total = df_total
        st.session_state.pre_confidence_avg = np.mean(pre_confidence)

        save_result({
            "participant_id": st.session_state.participant_id,
            "timestamp": datetime.now().isoformat(),
            "stage": "pre",
            "detection_score": score,
            "detection_total": total,
            "deepfake_score": df_score,
            "deepfake_total": df_total,
            "confidence_avg": np.mean(pre_confidence),
            "intent_avg": None,
            "prior_scam_exposure": st.session_state.prior_scam_exposure,
        })
        st.session_state.stage = "intervention"
        st.rerun()

# --- INTERVENTION ---
elif st.session_state.stage == "intervention":
    st.header("Deteksi Penipuan Audio Video menggunakan Sistem AI Terintegrasi")
    st.write("Inovasi pengubah perilaku pengguna menuju kebiasaan finansial yang lebih baik")

    # put a gif
    st.image("assets/manipulation.gif", caption="Gif 1. Perbandingan konten")    
    st.image("model/flow.png", caption="Gambar 1. Arsitektur model")
    st.image("assets/sistem.gif", caption="Gif 2. Video Asli & Audio Palsu")
    st.image("assets/sistem2.gif", caption="Gif 3. Video Palsu & Audio Palsu")

    st.info(
        "Fitur dari sistem kami:\n"
        "- Peringatan risiko real time\n"
        "- Deteksi potensi penipuan Video Audio (Deepfake Spoofing)\n"
    )
    if st.button("Lanjutkan ke Post-Test"):
        st.session_state.stage = "post_test"
        st.rerun()

# --- POST-TEST (AI-assisted) ---
elif st.session_state.stage == "post_test":
    st.header("Post-Test")
    st.caption(
        "Kali ini Anda akan dibantu oleh sistem AI kami saat menilai klip video/audio — "
        "ini mensimulasikan bagaimana produk akan benar-benar digunakan."
    )
    st.subheader("Part 1 — Deteksi Penipuan")
    post_answers = render_detection_quiz(SET_B, "post")

    st.subheader("Part 2 — Video / Audio (dibantu AI)")
    st.caption("Klip berbeda dari sebelumnya, dengan tingkat kesulitan yang setara.")
    post_df_answers = render_deepfake_quiz_assisted(DEEPFAKE_SET_B, "post_df")

    st.subheader("Part 3 — Tingkat Keyakinan")
    post_confidence = render_likert_block(CONFIDENCE_QUESTIONS, "post_conf")

    st.subheader("Part 4 — Niat Perilaku")
    post_intent = render_likert_block(INTENT_QUESTIONS, "post_intent")

    if st.button("Submit Post-Test"):
        score, total = score_detection(post_answers, SET_B)
        df_score, df_total = score_deepfake_quiz(post_df_answers, DEEPFAKE_SET_B)
        st.session_state.post_score = score
        st.session_state.post_total = total
        st.session_state.post_df_score = df_score
        st.session_state.post_df_total = df_total
        st.session_state.post_confidence_avg = np.mean(post_confidence)
        st.session_state.post_intent_avg = np.mean(post_intent)

        save_result({
            "participant_id": st.session_state.participant_id,
            "timestamp": datetime.now().isoformat(),
            "stage": "post",
            "detection_score": score,
            "detection_total": total,
            "deepfake_score": df_score,
            "deepfake_total": df_total,
            "confidence_avg": np.mean(post_confidence),
            "intent_avg": np.mean(post_intent),
            "prior_scam_exposure": st.session_state.prior_scam_exposure,
        })
        st.session_state.stage = "results"
        st.rerun()

# --- RESULTS ---
elif st.session_state.stage == "results":
    st.header("Hasil Anda")
    st.caption(
        "Catatan: skor Video/Audio pre-test diukur TANPA bantuan AI, sedangkan "
        "post-test diukur DENGAN bantuan AI — perbandingan ini menunjukkan seberapa "
        "besar produk membantu, bukan murni peningkatan kemampuan manusia."
    )

    pre_pct = st.session_state.pre_score / st.session_state.pre_total * 100
    post_pct = st.session_state.post_score / st.session_state.post_total * 100
    df_pre_pct = st.session_state.pre_df_score / st.session_state.pre_df_total * 100
    df_post_pct = st.session_state.post_df_score / st.session_state.post_df_total * 100

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Deteksi penipuan (pre)", f"{st.session_state.pre_score}/{st.session_state.pre_total}",
                   f"{pre_pct:.0f}%")
    with col2:
        st.metric("Deteksi penipuan (post)", f"{st.session_state.post_score}/{st.session_state.post_total}",
                   f"{post_pct - pre_pct:+.0f}pp")

    col3, col4 = st.columns(2)
    with col3:
        st.metric("Deteksi deepfake (pre, mandiri)",
                   f"{st.session_state.pre_df_score}/{st.session_state.pre_df_total}",
                   f"{df_pre_pct:.0f}%")
    with col4:
        st.metric("Deteksi deepfake (post, dibantu AI)",
                   f"{st.session_state.post_df_score}/{st.session_state.post_df_total}",
                   f"{df_post_pct - df_pre_pct:+.0f}pp")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(["Pre", "Post"], [pre_pct, post_pct], color=["#888", "#2b7"])
    axes[0].set_ylabel("Akurasi (%)"); axes[0].set_ylim(0, 100)
    axes[0].set_title("Deteksi Penipuan")

    axes[1].bar(["Pre\n(mandiri)", "Post\n(dibantu AI)"], [df_pre_pct, df_post_pct], color=["#888", "#2b7"])
    axes[1].set_ylabel("Akurasi (%)"); axes[1].set_ylim(0, 100)
    axes[1].set_title("Deteksi Video/Audio Palsu")

    axes[2].bar(["Pre", "Post"], [st.session_state.pre_confidence_avg, st.session_state.post_confidence_avg],
                color=["#888", "#2b7"])
    axes[2].set_ylabel("Rata-rata keyakinan (1-5)"); axes[2].set_ylim(0, 5)
    axes[2].set_title("Tingkat Keyakinan")

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Niat perilaku (hanya post-test)")
    st.write(f"Rata-rata skor niat: {st.session_state.post_intent_avg:.2f} / 5")

    st.divider()
    if st.button("Mulai peserta baru"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()
    st.subheader("Hasil agregat seluruh peserta")
    df = load_all_results()
    if len(df) > 0:
        
        pre_df_agg = df[df["stage"] == "pre"].copy()
        post_df_agg = df[df["stage"] == "post"].copy()

        if len(pre_df_agg) > 0 and len(post_df_agg) > 0:
            pre_df_agg["pct"] = pre_df_agg["detection_score"] / pre_df_agg["detection_total"] * 100
            post_df_agg["pct"] = post_df_agg["detection_score"] / post_df_agg["detection_total"] * 100
            pre_df_agg["df_pct"] = pre_df_agg["deepfake_score"] / pre_df_agg["deepfake_total"] * 100
            post_df_agg["df_pct"] = post_df_agg["deepfake_score"] / post_df_agg["deepfake_total"] * 100

            st.write(f"N = {len(pre_df_agg)} peserta (pre), {len(post_df_agg)} (post)")

            fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
            axes2[0].bar(["Pre (avg)", "Post (avg)"],
                            [pre_df_agg["pct"].mean(), post_df_agg["pct"].mean()], color=["#888", "#2b7"])
            axes2[0].set_ylabel("Rata-rata akurasi (%)"); axes2[0].set_ylim(0, 100)
            axes2[0].set_title("Deteksi Penipuan (Agregat)")

            axes2[1].bar(["Pre\n(mandiri)", "Post\n(dibantu AI)"],
                            [pre_df_agg["df_pct"].mean(), post_df_agg["df_pct"].mean()], color=["#888", "#2b7"])
            axes2[1].set_ylabel("Rata-rata akurasi (%)"); axes2[1].set_ylim(0, 100)
            axes2[1].set_title("Deteksi Deepfake (Agregat)")
            plt.tight_layout()
            st.pyplot(fig2)

            from scipy import stats
            merged = pre_df_agg.merge(post_df_agg, on="participant_id", suffixes=("_pre", "_post"))
            if len(merged) >= 2:
                t_stat, p_val = stats.ttest_rel(merged["pct_pre"], merged["pct_post"])
                st.write(f"Uji-t berpasangan (deteksi penipuan): t={t_stat:.3f}, p={p_val:.4f} (n={len(merged)})")
                if p_val < 0.05:
                    st.success("Peningkatan signifikan secara statistik (p < 0.05).")
                else:
                    st.warning("Belum signifikan secara statistik pada ukuran sampel ini.")

                t_stat_df, p_val_df = stats.ttest_rel(merged["df_pct_pre"], merged["df_pct_post"])
                st.write(f"Uji-t berpasangan (deteksi deepfake, efek bantuan AI): "
                            f"t={t_stat_df:.3f}, p={p_val_df:.4f} (n={len(merged)})")
                if p_val_df < 0.05:
                    st.success("Bantuan AI meningkatkan akurasi deteksi secara signifikan (p < 0.05).")
                else:
                    st.warning("Efek bantuan AI belum signifikan secara statistik pada ukuran sampel ini.")
        else:
            st.info("Perlu minimal satu respons pre dan post lengkap untuk menampilkan perbandingan agregat.")

        with st.expander("Data mentah"):
            st.dataframe(df)
    else:
        st.info("Belum ada data.")