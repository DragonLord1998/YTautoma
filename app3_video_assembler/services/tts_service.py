"""
TTS Service
Text-to-speech with multiple backends:
- Edge-TTS (simple, always works, no GPU)
- VibeVoice (advanced, requires setup)
"""

import subprocess
import asyncio
import os
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import VIBEVOICE_MODEL, VIBEVOICE_SPEAKER, VIBEVOICE_REPO_PATH


class EdgeTTSService:
    """Simple TTS using Microsoft Edge TTS (no GPU, always works)"""
    
    # Available voices
    VOICES = {
        "en-US": ["en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural"],
        "en-GB": ["en-GB-RyanNeural", "en-GB-SoniaNeural"],
    }
    
    def __init__(self, voice: str = "en-US-GuyNeural"):
        self.voice = voice
        self.available = self._check_available()
    
    def _check_available(self) -> bool:
        """Check if edge-tts is installed"""
        try:
            import edge_tts
            return True
        except ImportError:
            print("⚠️ edge-tts not installed. Install with: pip install edge-tts")
            return False
    
    def synthesize(self, text: str, output_path: Path, voice: Optional[str] = None) -> Path:
        """Generate speech from text"""
        if not self.available:
            raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")
        
        output_path = Path(output_path).with_suffix(".mp3")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        voice = voice or self.voice
        
        print(f"🔊 Generating speech with Edge-TTS...")
        print(f"   Voice: {voice}")
        print(f"   Text: {text[:50]}...")
        
        # Run async function
        async def generate():
            import edge_tts
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(output_path))
        
        asyncio.run(generate())
        
        print(f"✅ Audio saved: {output_path}")
        return output_path
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration using ffprobe"""
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())


class VibeVoiceTTSService:
    """Advanced TTS using Microsoft VibeVoice (requires setup)"""
    
    SPEAKERS = ["Carter", "Evelyn", "Andrew", "Aria"]
    
    def __init__(
        self,
        repo_path: str = VIBEVOICE_REPO_PATH,
        model_id: str = VIBEVOICE_MODEL,
        speaker: str = VIBEVOICE_SPEAKER
    ):
        self.repo_path = Path(repo_path)
        self.model_id = model_id
        self.speaker = speaker
        self._check_installation()
    
    def _check_installation(self):
        """Check if VibeVoice is installed"""
        self.available = False
        
        if not self.repo_path.exists():
            print(f"⚠️ VibeVoice not found at {self.repo_path}")
            return
        
        # Check if vibevoice module is importable
        try:
            import vibevoice
            self.available = True
            print(f"✅ VibeVoice ready")
        except ImportError:
            print("⚠️ VibeVoice module not installed. Run: cd models/VibeVoice && pip install -e .")
    
    def synthesize(self, text: str, output_path: Path, speaker: Optional[str] = None) -> Path:
        """Generate speech using VibeVoice"""
        if not self.available:
            raise RuntimeError("VibeVoice not available")
        
        speaker = speaker or self.speaker
        output_path = Path(output_path).with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write text to temp file
        temp_txt = output_path.parent / "temp_input.txt"
        temp_txt.write_text(text)
        
        cmd = [
            sys.executable,
            str(self.repo_path / "demo" / "realtime_model_inference_from_file.py"),
            "--model_path", self.model_id,
            "--txt_path", str(temp_txt),
            "--speaker_name", speaker,
            "--output_path", str(output_path)
        ]
        
        print(f"🔊 Generating speech with VibeVoice...")
        
        try:
            result = subprocess.run(cmd, cwd=str(self.repo_path), 
                                   capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(f"VibeVoice failed: {result.stderr[-200:]}")
            return output_path
        finally:
            temp_txt.unlink(missing_ok=True)
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file"""
        import wave
        try:
            with wave.open(str(audio_path), 'rb') as wf:
                return wf.getnframes() / float(wf.getframerate())
        except:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())


class TTSFactory:
    """Factory for creating TTS service"""
    
    @staticmethod
    def create(preferred: str = "edge"):
        """
        Create TTS service.
        
        Args:
            preferred: 'edge' (simple), 'vibevoice' (advanced)
        """
        # Check env variable
        preferred = os.getenv("TTS_ENGINE", preferred).lower()
        
        if preferred == "edge":
            service = EdgeTTSService()
            if service.available:
                return service
        
        if preferred == "vibevoice":
            service = VibeVoiceTTSService()
            if service.available:
                return service
            print("Falling back to Edge-TTS...")
        
        # Default fallback
        service = EdgeTTSService()
        if service.available:
            return service
        
        raise RuntimeError(
            "No TTS available. Install edge-tts: pip install edge-tts"
        )
