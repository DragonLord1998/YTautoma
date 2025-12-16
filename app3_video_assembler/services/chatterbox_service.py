"""
ChatterBox TTS Service
High-quality text-to-speech using Resemble AI's ChatterBox.
https://github.com/resemble-ai/chatterbox
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import CHATTERBOX_DEVICE, CHATTERBOX_VOICE_REF


class ChatterboxTTSService:
    """High-quality TTS using ChatterBox from Resemble AI"""
    
    def __init__(
        self,
        device: str = CHATTERBOX_DEVICE,
        voice_ref: Optional[str] = CHATTERBOX_VOICE_REF
    ):
        self.device = device
        self.voice_ref = Path(voice_ref) if voice_ref else None
        self.available = self._check_installation()
        self.model = None
        self.sr = 24000  # Default sample rate
    
    def _check_installation(self) -> bool:
        """Check if ChatterBox is installed"""
        try:
            import torchaudio
            from chatterbox.tts import ChatterboxTTS
            return True
        except ImportError as e:
            print(f"⚠️ ChatterBox not installed: {e}")
            print("   Install with: pip install chatterbox-tts")
            return False
    
    def load_model(self):
        """Load ChatterBox model (call once, reuse for multiple generations)"""
        if self.model is not None:
            return
        
        if not self.available:
            raise RuntimeError("ChatterBox not installed")
        
        print(f"🔄 Loading ChatterBox model on {self.device}...")
        from chatterbox.tts import ChatterboxTTS
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sr = self.model.sr
        print("✅ ChatterBox loaded")
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("🗑️ ChatterBox unloaded")
    
    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_ref: Optional[Path] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5
    ) -> Path:
        """
        Generate speech from text using ChatterBox.
        
        Args:
            text: Text to synthesize
            output_path: Output audio file path
            voice_ref: Optional reference audio for voice cloning
            exaggeration: Expression level (0.0-1.0, higher = more dramatic)
            cfg_weight: CFG weight (lower = slower, more deliberate pacing)
            
        Returns:
            Path to generated audio file
        """
        if not self.available:
            raise RuntimeError("ChatterBox not installed")
        
        self.load_model()
        
        output_path = Path(output_path).with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        voice = voice_ref or self.voice_ref
        
        print(f"🔊 Generating speech with ChatterBox...")
        print(f"   Text: {text[:60]}...")
        
        import torchaudio
        
        # Generate audio
        if voice and voice.exists():
            print(f"   Voice ref: {voice.name}")
            wav = self.model.generate(
                text,
                audio_prompt_path=str(voice),
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
        else:
            wav = self.model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
        
        # Save audio
        torchaudio.save(str(output_path), wav, self.sr)
        
        print(f"✅ Audio saved: {output_path}")
        return output_path
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file in seconds"""
        try:
            import torchaudio
            info = torchaudio.info(str(audio_path))
            return info.num_frames / info.sample_rate
        except:
            # Fallback to ffprobe
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())


class ChatterboxTurboTTSService(ChatterboxTTSService):
    """
    Faster TTS using ChatterBox Turbo model.
    Supports paralinguistic tags like [chuckle], [sigh], etc.
    """
    
    def load_model(self):
        """Load ChatterBox Turbo model"""
        if self.model is not None:
            return
        
        if not self.available:
            raise RuntimeError("ChatterBox not installed")
        
        print(f"🔄 Loading ChatterBox Turbo on {self.device}...")
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            self.sr = self.model.sr
            print("✅ ChatterBox Turbo loaded")
        except ImportError:
            # Fall back to regular ChatterBox
            print("⚠️ Turbo not available, using standard ChatterBox")
            from chatterbox.tts import ChatterboxTTS
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            self.sr = self.model.sr
