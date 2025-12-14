# YouTube Shorts Automation (Fully Offline)

Automated 60-second YouTube Shorts using **fully local AI models**:

| Component | Model | Source |
|-----------|-------|--------|
| Story | Gemma 3 27B-abliterated | Ollama |
| Images | Z-Image-Turbo | Diffusers |
| Consistency | Qwen-Image-Edit | Transformers |
| Video | Wan 2.2 I2V | Local CLI |
| Voice | VibeVoice-Realtime | Local CLI |

## Requirements

- **GPU**: RTX 4090 (24GB) or better
- **RAM**: 32GB+
- **Storage**: 100GB+ for models
- **OS**: Linux recommended (Docker for VibeVoice)

## Setup

### 1. Install Ollama + Gemma 3

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Gemma 3 27B abliterated
ollama pull gemma3:27b-abliterated
ollama serve
```

### 2. Clone Repositories

```bash
cd ~/models

# Z-Image (already in diffusers)
pip install git+https://github.com/huggingface/diffusers

# Wan 2.2
git clone https://github.com/Wan-Video/Wan2.2.git
pip install -r Wan2.2/requirements.txt

# Download Wan 2.2 I2V model
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B

# VibeVoice
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice && pip install -e .
```

### 3. Install FFmpeg

```bash
# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your model paths
```

### 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Full pipeline (random topic)
python main.py

# With category
python main.py -c mystery
python main.py -c horror
python main.py -c sci-fi

# Custom topic
python main.py -t "A robot discovers emotions"

# Story only (fast test)
python main.py --story-only

# Images only (no Wan 2.2 video - faster)
python main.py --images-only
```

## Project Structure

```
Skarg/
├── main.py                      # Orchestrator
├── app1_story_generator/        # Gemma 3 27B via Ollama
├── app2_visual_generator/       # Z-Image + Wan 2.2
├── app3_video_assembler/        # VibeVoice + FFmpeg
├── shared/                      # Config & models
├── models/                      # Local model storage
└── output/                      # Generated content
```

## Memory Management

For GPUs with <48GB VRAM, models are loaded/unloaded sequentially:
1. Load Z-Image → Generate all images → Unload
2. Load Wan 2.2 → Generate all videos → Unload
3. Load VibeVoice → Generate all audio → Assemble

Enable `LOW_VRAM_MODE=true` in `.env` for automatic memory management.
