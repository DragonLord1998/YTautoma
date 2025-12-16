# Shared Configuration - Local Offline Models
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
ASSETS_DIR = PROJECT_ROOT / "app3_video_assembler" / "assets"
MODELS_DIR = PROJECT_ROOT / "models"  # Local model storage

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ===========================================
# Ollama Config (Local LLM for Story)
# ===========================================
# DeepSeek V3 Q4 quantized: ~50GB, 671B params (37B active via MoE)
# Alternative: qwen2.5:72b (~42GB), llama3.1:70b (~40GB)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-v3:q4_k_m")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ===========================================
# Z-Image Config (Local Diffusers)
# ===========================================
ZIMAGE_MODEL = os.getenv("ZIMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
ZIMAGE_DEVICE = os.getenv("ZIMAGE_DEVICE", "cuda")

# ===========================================
# Wan 2.2 Config (Local Video Generation)
# ===========================================
WAN_REPO_PATH = os.getenv("WAN_REPO_PATH", str(MODELS_DIR / "Wan2.2"))
WAN_MODEL_PATH = os.getenv("WAN_MODEL_PATH", str(MODELS_DIR / "Wan2.2-I2V-A14B"))
WAN_T5_CPU = os.getenv("WAN_T5_CPU", "true").lower() == "true"  # Offload T5 to CPU
WAN_OFFLOAD_MODEL = os.getenv("WAN_OFFLOAD_MODEL", "true").lower() == "true"

# ===========================================
# Qwen-Image-Edit Config (Local)
# ===========================================
QWEN_IMAGE_EDIT_MODEL = os.getenv("QWEN_IMAGE_EDIT_MODEL", "Qwen/Qwen-Image-Edit")

# ===========================================
# VibeVoice TTS Config (Local)
# ===========================================
VIBEVOICE_MODEL = os.getenv("VIBEVOICE_MODEL", "microsoft/VibeVoice-Realtime-0.5B")
VIBEVOICE_SPEAKER = os.getenv("VIBEVOICE_SPEAKER", "Carter")
VIBEVOICE_REPO_PATH = os.getenv("VIBEVOICE_REPO_PATH", str(MODELS_DIR / "VibeVoice"))

# ===========================================
# Long-Form Video Settings (20 x 1-minute parts)
# ===========================================
TOTAL_PARTS = int(os.getenv("TOTAL_PARTS", "20"))
PART_DURATION = int(os.getenv("PART_DURATION", "60"))  # 1 minute each
TARGET_DURATION = TOTAL_PARTS * PART_DURATION  # 1200 seconds = 20 minutes
SCENES_PER_PART = int(os.getenv("SCENES_PER_PART", "5"))

# Legacy short-form settings (for backwards compatibility)
SCENES_COUNT = 6  # Original shorts mode
SHORT_DURATION = 60  # Original 60-second shorts

# ===========================================
# Video Settings
# ===========================================
VIDEO_WIDTH = 1280   # 720p horizontal for long-form
VIDEO_HEIGHT = 720   # Standard YouTube format
VIDEO_FPS = 24

# Wan 2.2 video settings (14B model for 720p)
WAN_VIDEO_SIZE = os.getenv("WAN_VIDEO_SIZE", "1280*720")  # 720p horizontal
WAN_VIDEO_FRAMES = int(os.getenv("WAN_VIDEO_FRAMES", "241"))  # ~10 seconds at 24fps
WAN_TASK = os.getenv("WAN_TASK", "i2v-14B")  # Full 14B model

# ===========================================
# Research Agent Config
# ===========================================
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "qwen3:32b")  # 36B+ for deep research
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")

# ===========================================
# ChatterBox TTS Config
# ===========================================
CHATTERBOX_DEVICE = os.getenv("CHATTERBOX_DEVICE", "cuda")
CHATTERBOX_VOICE_REF = os.getenv("CHATTERBOX_VOICE_REF", None)  # Optional reference audio

# ===========================================
# Hardware Settings
# ===========================================
TORCH_DTYPE = "bfloat16"  # or "float16"
LOW_VRAM_MODE = os.getenv("LOW_VRAM_MODE", "true").lower() == "true"
