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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:27b-abliterated")
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
# Video Settings
# ===========================================
VIDEO_WIDTH = 1088   # Must be divisible by 16 for Z-Image
VIDEO_HEIGHT = 1920  # 9:16 vertical for Shorts
VIDEO_FPS = 24
TARGET_DURATION = 60  # seconds
SCENES_COUNT = 6
SCENE_DURATION = TARGET_DURATION // SCENES_COUNT  # ~10 seconds each

# Wan 2.2 video settings
WAN_VIDEO_SIZE = "480*848"  # Vertical 480p (faster generation)
WAN_VIDEO_FRAMES = 121  # ~5 seconds at 24fps

# ===========================================
# Hardware Settings
# ===========================================
TORCH_DTYPE = "bfloat16"  # or "float16"
LOW_VRAM_MODE = os.getenv("LOW_VRAM_MODE", "true").lower() == "true"
