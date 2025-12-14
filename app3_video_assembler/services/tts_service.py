"""
VibeVoice TTS Service (Local)
Text-to-speech using Microsoft VibeVoice-Realtime.
https://github.com/microsoft/VibeVoice
"""

import subprocess
import os
from pathlib import Path
from typing import Optional
import sys
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import VIBEVOICE_MODEL, VIBEVOICE_SPEAKER, VIBEVOICE_REPO_PATH


class VibeVoiceTTSService:
    """Generate speech using local VibeVoice-Realtime model"""
    
    # Available speakers
    SPEAKERS = {
        "en": ["Carter", "Evelyn", "Andrew", "Aria"],
        "zh": ["Zhiyu", "Yunxi"],
        "de": ["Amala"],
        "fr": ["Alain"],
        "jp": ["Nanami"],
        "kr": ["Jin-ae"],
    }
    
    def __init__(
        self,
        repo_path: str = VIBEVOICE_REPO_PATH,
        model_id: str = VIBEVOICE_MODEL,
        speaker: str = VIBEVOICE_SPEAKER
    ):
        """
        Initialize VibeVoice service.
        
        Args:
            repo_path: Path to cloned VibeVoice repository
            model_id: HuggingFace model ID
            speaker: Speaker name (e.g., 'Carter', 'Evelyn')
        """
        self.repo_path = Path(repo_path)
        self.model_id = model_id
        self.speaker = speaker
        self._check_installation()
    
    def _check_installation(self):
        """Check if VibeVoice is properly installed"""
        self.available = False
        
        if not self.repo_path.exists():
            print(f"⚠️ VibeVoice repo not found at {self.repo_path}")
            print("   Clone with: git clone https://github.com/microsoft/VibeVoice.git")
            return
        
        inference_script = self.repo_path / "demo" / "realtime_model_inference_from_file.py"
        if not inference_script.exists():
            print(f"⚠️ VibeVoice inference script not found")
            return
        
        self.available = True
        print(f"✅ VibeVoice ready at {self.repo_path}")
    
    def synthesize(
        self,
        text: str,
        output_path: Path,
        speaker: Optional[str] = None
    ) -> Path:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to speak
            output_path: Output audio file path
            speaker: Speaker name override
            
        Returns:
            Path to generated audio file
        """
        if not self.available:
            raise RuntimeError(
                "VibeVoice not available. Please install:\n"
                "1. git clone https://github.com/microsoft/VibeVoice.git\n"
                "2. cd VibeVoice && pip install -e ."
            )
        
        speaker = speaker or self.speaker
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write text to temp file
        temp_txt = output_path.parent / "temp_input.txt"
        temp_txt.write_text(text)
        
        # Run inference
        cmd = [
            sys.executable,
            str(self.repo_path / "demo" / "realtime_model_inference_from_file.py"),
            "--model_path", self.model_id,
            "--txt_path", str(temp_txt),
            "--speaker_name", speaker,
            "--output_path", str(output_path.with_suffix(".wav"))
        ]
        
        print(f"🔊 Generating speech with VibeVoice...")
        print(f"   Speaker: {speaker}")
        print(f"   Text: {text[:50]}...")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                print(f"❌ VibeVoice error:\n{result.stderr[-500:]}")
                raise RuntimeError(f"TTS failed: {result.stderr[-200:]}")
            
            # Find output file
            wav_output = output_path.with_suffix(".wav")
            if wav_output.exists():
                print(f"✅ Audio generated: {wav_output}")
                return wav_output
            
            # Check for output in different location
            possible_outputs = list(self.repo_path.glob("output/*.wav"))
            if possible_outputs:
                latest = max(possible_outputs, key=lambda p: p.stat().st_mtime)
                shutil.move(str(latest), str(wav_output))
                return wav_output
            
            raise RuntimeError("No audio file generated")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("TTS timed out")
        finally:
            temp_txt.unlink(missing_ok=True)
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file in seconds"""
        import wave
        try:
            with wave.open(str(audio_path), 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
        except:
            # Fallback using ffprobe
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())


class VibeVoicePythonService:
    """
    Alternative VibeVoice service using Python API directly.
    More flexible but requires model to be loaded in memory.
    """
    
    def __init__(self, model_id: str = VIBEVOICE_MODEL, speaker: str = VIBEVOICE_SPEAKER):
        self.model_id = model_id
        self.speaker = speaker
        self.model = None
        self._loaded = False
    
    def load_model(self):
        """Load VibeVoice model"""
        if self._loaded:
            return
        
        print(f"🔄 Loading VibeVoice model: {self.model_id}")
        
        try:
            from vibevoice import VibeVoiceRealtime
            
            self.model = VibeVoiceRealtime.from_pretrained(self.model_id)
            self.model = self.model.cuda()
            
            self._loaded = True
            print("✅ VibeVoice loaded")
            
        except ImportError:
            print("⚠️ VibeVoice package not installed")
            print("   Install with: pip install -e /path/to/VibeVoice")
            raise
    
    def synthesize(self, text: str, output_path: Path, speaker: Optional[str] = None) -> Path:
        """Synthesize using Python API"""
        if not self._loaded:
            self.load_model()
        
        import soundfile as sf
        
        speaker = speaker or self.speaker
        output_path = Path(output_path).with_suffix(".wav")
        
        print(f"🔊 Generating speech...")
        
        audio = self.model.tts(text, speaker_name=speaker)
        sf.write(str(output_path), audio, 24000)
        
        print(f"✅ Audio saved: {output_path}")
        return output_path


class TTSFactory:
    """Factory for creating TTS service based on availability"""
    
    @staticmethod
    def create(preferred: str = "vibevoice"):
        """
        Create TTS service with fallback options.
        
        Args:
            preferred: 'vibevoice', 'piper', or 'coqui'
        """
        # Try VibeVoice first
        if preferred.lower() == "vibevoice":
            service = VibeVoiceTTSService()
            if service.available:
                return service
            print("Falling back to Piper...")
        
        # Try Piper
        try:
            from .tts_service_piper import PiperTTSService
            service = PiperTTSService()
            if service.available:
                return service
        except:
            pass
        
        raise RuntimeError(
            "No TTS engine available. Please install VibeVoice:\n"
            "git clone https://github.com/microsoft/VibeVoice.git\n"
            "cd VibeVoice && pip install -e ."
        )
