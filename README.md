## 🤖 Powered by Tarteel AI

This project uses the open-source [Whisper model fine-tuned for Arabic Quran](https://huggingface.co/tarteel-ai/whisper-base-ar-quran) provided by **Tarteel AI** via [Hugging Face](https://huggingface.co/tarteel-ai). Huge thanks to Tarteel for enabling this innovation in Quranic voice technology.

> **Model**: `tarteel-ai/whisper-base-ar-quran`

---

## 🚀 Quick Start

### 1. Install Requirements

> Requires: Python 3.8+ and [FFmpeg](https://ffmpeg.org/download.html) installed and in your system path.

```bash
git clone https://github.com/yourusername/quran-matcher-api.git
cd quran-matcher-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# to run api:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
