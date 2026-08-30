# Audio Video Deepfake Detection
 Prototype for detecting scam/fraud relevant to digital banking, focused on video and audio deepfake detection with a human in evaluation layer.

![flow of model](assets/flow.png)

## Problem

Social engineering scams (phishing, OTP fraud, impersonation calls, QRIS fraud, and increasingly AI-generated deepfake video/audio) are a threat in digital banking. Static awareness campaigns underperform because victims are targeted at moments of urgency, not knowledge gaps. This project explores a real time detection layer as a complement to user education.

## Components

- **Video deepfake detection** — EfficientNet-B0 backbone with CBAM (channel + spatial attention) and a bidirectional GRU for temporal modeling across consecutive frames.
- **Audio spoof detection** — CNN on mel-spectrograms, trained on combined Indonesian (SEA-Spoof) and re-encoded/synced (BioDeepAV) audio sources.
- **Combined inference** — video and audio models run independently, outputs are merged using an extreme-signal-priority rule (a confident signal from either modality is trusted directly; otherwise a weighted average is used).
- **Live test app** — Streamlit interface for uploading a video and getting a combined video + audio verdict.
- **Pre/post evaluation study** — Streamlit survey measuring human scam detection accuracy before and after using the tool, including an AI-assisted deepfake detection comparison.

## Datasets

| Source | Modality | Notes |
|---|---|---|
| FaceForensics++ (c23) | Video | 6 manipulation methods + original |
| DFDC (sample set) | Video | Real class limited to 77 (sample-set constraint) |
| DeeperForensics-1.0 | Video | Identity/lighting/emotion/camera combinatorial real set |
| BioDeepAV | Video + Audio | Paired real/fake video and audio |
| SEA-Spoof (Indonesian subset) | Audio | TTS/voice-conversion spoofing |
| Celeb-DF v2 | Video |  Used only for final generalization testing |

## Data pipeline

1. Selective download via Kaggle API and Hugging Face Hub.
2. Consecutive-frame extraction to preserve temporal continuity for the GRU
3. Face detection/cropping via MediaPipe, with nearest-valid-frame gap filling to avoid breaking sequence continuity
4. Video-level train/val/test splitting to prevent data leakage across sets.
5. Class-weighted loss and weighted sampling to address source/class imbalance.

## Model results

| Evaluation | Accuracy | AUC | Notes |
|---|---|---|---|
| In-distribution (val) | 94.6% | 0.991 | FF++, DFDC, DeeperForensics, partial BioDeepAV |
| BioDeepAV held-out (40%, partial-exposure) | 97.4% | 0.997 | Same source partially seen in training |
| Celeb-DF v2 (fully unseen source) | 70.5% → 71.8%* | 0.788 → 0.806* | After augmentation fine-tune targeting compression/quality robustness |


## Repository structure

```
assets/
live_test/
model/
  video/   # CBAM+GRU model checkpoints
  audio/               # audio spoof model checkpoints
app.py            # live video+audio inference demo
question.py      # human evaluation study (pre/post, AI-assisted)
```

## Running the live demo

```bash
pip install streamlit torch timm albumentations mediapipe opencv-python librosa soundfile
streamlit run app.py
```

## Running the evaluation survey

```bash
streamlit run question.py
```
## Documentation

![Comparison Deepfake](assets/manipulation.gif)
![App interface](assets/sistem.gif)
![App interface 2](assets/sistem2.gif)