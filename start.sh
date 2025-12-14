#!/bin/bash
# ===========================================
# YTautoma Server Setup & Start Script
# ===========================================
# Usage: ./start.sh [setup|run|story-only]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[YTautoma]${NC} $1"; }
warn() { echo -e "${YELLOW}[Warning]${NC} $1"; }
error() { echo -e "${RED}[Error]${NC} $1"; exit 1; }

# ===========================================
# SETUP FUNCTION
# ===========================================
setup() {
    log "Starting YTautoma setup..."
    
    # Create models directory
    mkdir -p "${MODELS_DIR}"
    
    # 1. Install Python dependencies
    log "Installing Python dependencies..."
    pip install -r "${SCRIPT_DIR}/requirements.txt"
    pip install git+https://github.com/huggingface/diffusers
    
    # 2. Clone Wan 2.2
    if [ ! -d "${MODELS_DIR}/Wan2.2" ]; then
        log "Cloning Wan 2.2..."
        git clone https://github.com/Wan-Video/Wan2.2.git "${MODELS_DIR}/Wan2.2"
        pip install -r "${MODELS_DIR}/Wan2.2/requirements.txt"
    else
        warn "Wan 2.2 already exists, skipping..."
    fi
    
    # 3. Download Wan 2.2 I2V model
    if [ ! -d "${MODELS_DIR}/Wan2.2-I2V-A14B" ]; then
        log "Downloading Wan 2.2 I2V model (~50GB)..."
        pip install "huggingface_hub[cli]"
        huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir "${MODELS_DIR}/Wan2.2-I2V-A14B"
    else
        warn "Wan 2.2 model already exists, skipping..."
    fi
    
    # 4. Clone VibeVoice
    if [ ! -d "${MODELS_DIR}/VibeVoice" ]; then
        log "Cloning VibeVoice..."
        git clone https://github.com/microsoft/VibeVoice.git "${MODELS_DIR}/VibeVoice"
        cd "${MODELS_DIR}/VibeVoice" && pip install -e . && cd "${SCRIPT_DIR}"
    else
        warn "VibeVoice already exists, skipping..."
    fi
    
    # 5. Setup Ollama
    if ! command -v ollama &> /dev/null; then
        log "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    
    # 6. Pull Gemma 3
    log "Pulling Gemma 3 27B-abliterated..."
    ollama pull gemma3:27b-abliterated
    
    # 7. Create .env if not exists
    if [ ! -f "${SCRIPT_DIR}/.env" ]; then
        log "Creating .env configuration..."
        cat > "${SCRIPT_DIR}/.env" << EOF
# Ollama
OLLAMA_MODEL=gemma3:27b-abliterated
OLLAMA_BASE_URL=http://localhost:11434

# Z-Image
ZIMAGE_MODEL=Tongyi-MAI/Z-Image-Turbo
ZIMAGE_DEVICE=cuda

# Wan 2.2
WAN_REPO_PATH=${MODELS_DIR}/Wan2.2
WAN_MODEL_PATH=${MODELS_DIR}/Wan2.2-I2V-A14B
WAN_T5_CPU=true
WAN_OFFLOAD_MODEL=true

# VibeVoice
VIBEVOICE_REPO_PATH=${MODELS_DIR}/VibeVoice
VIBEVOICE_MODEL=microsoft/VibeVoice-Realtime-0.5B
VIBEVOICE_SPEAKER=Carter

# Hardware
LOW_VRAM_MODE=true
TORCH_DTYPE=bfloat16
EOF
    fi
    
    log "✅ Setup complete!"
    log "Run './start.sh run' to generate a video"
}

# ===========================================
# RUN FUNCTION
# ===========================================
run() {
    log "Starting YTautoma pipeline..."
    
    # Check if Ollama is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log "Starting Ollama server..."
        ollama serve &
        sleep 5
    fi
    
    # Run the pipeline
    cd "${SCRIPT_DIR}"
    python main.py "$@"
}

# ===========================================
# MAIN
# ===========================================
case "${1:-run}" in
    setup)
        setup
        ;;
    run)
        shift 2>/dev/null || true
        run "$@"
        ;;
    story-only)
        run --story-only
        ;;
    images-only)
        run --images-only
        ;;
    help|--help|-h)
        echo "YTautoma - YouTube Shorts Automation"
        echo ""
        echo "Usage: ./start.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  setup        Install all dependencies and models"
        echo "  run          Run the full pipeline (default)"
        echo "  story-only   Generate story only (fast test)"
        echo "  images-only  Generate story + images (no video)"
        echo ""
        echo "Options (for run):"
        echo "  -c CATEGORY  Story category (mystery, horror, sci-fi, etc.)"
        echo "  -t TOPIC     Custom story topic"
        echo ""
        echo "Examples:"
        echo "  ./start.sh setup"
        echo "  ./start.sh run -c mystery"
        echo "  ./start.sh run -t 'A robot learns to love'"
        ;;
    *)
        run "$@"
        ;;
esac
