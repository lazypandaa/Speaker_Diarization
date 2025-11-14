# 🎙️ Target Speaker Diarization & ASR System

A lightweight, fully offline pipeline for identifying a target speaker, performing speech segmentation, transcription, and generating structured per-speaker output. This repository contains a minimal, clean, and production-ready implementation ideal for hackathon submissions or research prototypes.

---

## 🚀 Features

* **Offline Target Speaker Identification** using Resemblyzer embeddings
* **Speech Segmentation** via WebRTC-VAD (no torchaudio issues)
* **Cosine Similarity Classification** (Target vs Speaker_B)
* **Whisper-based ASR** for full mixture transcription
* **Structured diarization.json** output with timestamps, speaker labels, confidence scores, and text
* **Generated target_speaker.wav** containing only the target speaker's voice
* Minimal dependencies & Mac-friendly

---

## 📂 Project Structure

```
target_speaker_diarization/
│── main.py
│── utils/
│     ├── vad.py
│     ├── embedding.py
│     ├── asr.py
│
│── samples/
│     ├── mixture_audio.wav
│     └── target_sample.wav
│
└── outputs/
       ├── target_speaker.wav
       └── diarization.json
```

---

## 📦 Installation

```bash
pip install librosa soundfile webrtcvad resemblyzer openai-whisper numpy
```

---

## ▶️ Running the Pipeline

1. Place your audio files inside the `samples/` folder:

```
samples/
   mixture_audio.wav
   target_sample.wav
```

2. Run the program:

```bash
python main.py
```

3. Check the outputs:

```
outputs/
   target_speaker.wav
   diarization.json
```

---

## 🧠 How It Works

### 1️⃣ VAD (WebRTC)

Splits mixture audio into speech-only segments.

### 2️⃣ Speaker Embeddings (Resemblyzer)

Extracts 256-D embeddings from both target_sample and each segment.

### 3️⃣ Speaker Matching

Cosine similarity determines whether each segment belongs to the target.

### 4️⃣ ASR (Whisper Tiny)

Transcribes the entire mixture offline.

### 5️⃣ JSON Generation

Text is aligned proportionally to segments → diarization.json.

### 6️⃣ Target Extraction

All segments labeled "Target" are concatenated → target_speaker.wav.

---

## 📝 Example diarization.json

```json
[
  {
    "speaker": "Target",
    "start": 0.0,
    "end": 36.0,
    "text": "This is like perfect...",
    "confidence": 0.89
  },
  {
    "speaker": "Speaker_B",
    "start": 43.8,
    "end": 44.8,
    "text": "to or whatever",
    "confidence": 0.31
  }
]
```

---

## 📜 Brief Approach (150–200 words)

My solution implements a lightweight, fully offline Target Speaker Diarization and ASR pipeline designed for efficiency and rapid deployment. The system processes two inputs: a multi-speaker mixture and a short target speaker sample. First, audio-level Voice Activity Detection (VAD) is performed using WebRTC-VAD to segment the mixture into meaningful speech regions. For speaker identification, I extract fixed-length embeddings from both the target sample and each speech segment using the Resemblyzer voice encoder. Cosine similarity is computed between embeddings, and segments are classified as “Target” or “Speaker_B” based on a threshold.

Recognized target segments are concatenated to produce a clean `target_speaker.wav` output. In parallel, the entire mixture is transcribed using Whisper (tiny model for offline CPU inference). The transcript is proportionally aligned to speech segments to generate per-segment text. A structured `diarization.json` file is then produced containing speaker label, timestamps, text, and confidence score.

This approach prioritizes simplicity, modularity, and full offline compatibility while still delivering practical diarization and transcription results suitable for real-time extension in future versions.

---

## 🛠️ Tech Stack

* **Voice Embeddings:** Resemblyzer
* **VAD:** WebRTC-VAD
* **ASR:** Whisper Tiny (CPU)
* **Audio Processing:** Librosa, SoundFile
* **Output:** JSON + WAV

---

## 🔮 Future Improvements

* PyAnnote for higher-quality diarization
* Real-time streaming with WebSockets
* Front-end visualization for speaker timelines
* Improved transcript-to-segment alignment

---

## 🤝 Contributing

Pull requests and feature enhancements are welcome!

---

## 📄 License

MIT License
