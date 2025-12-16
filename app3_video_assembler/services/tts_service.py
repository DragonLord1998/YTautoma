"""
TTS Service
Text-to-speech with multiple backends:
- Edge-TTS (simple, always works, no GPU)
- VibeVoice (advanced, high quality, requires setup)
"""

import subprocess
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import VIBEVOICE_MODEL, VIBEVOICE_SPEAKER, VIBEVOICE_REPO_PATH


class EdgeTTSService:
    """Simple TTS using Microsoft Edge TTS (no GPU, always works)"""
    
    VOICES = {
        "en-US": ["en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural"],
        "en-GB": ["en-GB-RyanNeural", "en-GB-SoniaNeural"],
    }
    
    def __init__(self, voice: str = "en-US-GuyNeural"):
        self.voice = voice
        self.available = self._check_available()
    
    def _check_available(self) -> bool:
        try:
            import edge_tts
            return True
        except ImportError:
            print("⚠️ edge-tts not installed. Install with: pip install edge-tts")
            return False
    
    def synthesize(self, text: str, output_path: Path, voice: Optional[str] = None) -> Path:
        if not self.available:
            raise RuntimeError("edge-tts not installed")
        
        output_path = Path(output_path).with_suffix(".mp3")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        voice = voice or self.voice
        
        print(f"🔊 Generating speech with Edge-TTS...")
        print(f"   Voice: {voice}")
        print(f"   Text: {text[:50]}...")
        
        async def generate():
            import edge_tts
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(output_path))
        
        asyncio.run(generate())
        
        print(f"✅ Audio saved: {output_path}")
        return output_path
    
    def get_audio_duration(self, audio_path: Path) -> float:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())


class VibeVoiceTTSService:
    """High-quality TTS using Microsoft VibeVoice"""
    
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
        self.available = self._check_installation()
    
    def _check_installation(self) -> bool:
        """Check if VibeVoice is properly installed"""
        if not self.repo_path.exists():
            print(f"⚠️ VibeVoice repo not found at {self.repo_path}")
            print("   Clone with: git clone https://github.com/microsoft/VibeVoice.git models/VibeVoice")
            print("   Then: cd models/VibeVoice && pip install -e .")
            return False
        
        # Check if the inference script exists
        inference_script = self.repo_path / "demo" / "realtime_model_inference_from_file.py"
        if not inference_script.exists():
            print(f"⚠️ VibeVoice inference script not found")
            return False
        
        # Check if vibevoice module is installed
        try:
            # Add repo to path temporarily to check
            sys.path.insert(0, str(self.repo_path))
            import vibevoice
            sys.path.pop(0)
            print(f"✅ VibeVoice ready")
            return True
        except ImportError:
            # Try installing it
            print("⚠️ VibeVoice module not installed. Attempting to install...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-e", "."],
                    cwd=str(self.repo_path),
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    print("✅ VibeVoice installed successfully")
                    return True
                else:
                    print(f"⚠️ VibeVoice install failed: {result.stderr[:200]}")
                    return False
            except Exception as e:
                print(f"⚠️ Could not install VibeVoice: {e}")
                return False
    
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
        
        print(f"🔊 Generating speech with VibeVoice...")
        print(f"   Speaker: {speaker}")
        print(f"   Text: {text[:50]}...")
        
        # Build command
        cmd = [
            sys.executable,
            str(self.repo_path / "demo" / "realtime_model_inference_from_file.py"),
            "--model_path", self.model_id,
            "--txt_path", str(temp_txt),
            "--speaker_name", speaker,
            "--output_path", str(output_path)
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_path) + ":" + env.get("PYTHONPATH", "")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=180
            )
            
            if result.returncode != 0:
                print(f"❌ VibeVoice error:\n{result.stderr[-500:]}")
                raise RuntimeError(f"VibeVoice failed: {result.stderr[-200:]}")
            
            if not output_path.exists():
                raise RuntimeError("VibeVoice did not produce output file")
            
            print(f"✅ Audio saved: {output_path}")
            return output_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("VibeVoice timed out")
        finally:
            temp_txt.unlink(missing_ok=True)
    
    def get_audio_duration(self, audio_path: Path) -> float:
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
    def create(preferred: str = "chatterbox"):
        """
        Create TTS service.
        
        Args:
            preferred: 'chatterbox' (best quality), 'vibevoice', or 'edge' (simple)
        """
        # Check env variable
        preferred = os.getenv("TTS_ENGINE", preferred).lower()
        
        # Try ChatterBox first (best quality)
        if preferred == "chatterbox":
            try:
                from app3_video_assembler.services.chatterbox_service import ChatterboxTTSService
                service = ChatterboxTTSService()
                if service.available:
                    print("✅ Using ChatterBox TTS")
                    return service
            except ImportError:
                pass
            print("⚠️ ChatterBox not available, trying fallbacks...")
        
        if preferred == "vibevoice":
            service = VibeVoiceTTSService()
            if service.available:
                return service
            print("Falling back to Edge-TTS...")
        
        if preferred == "edge":
            service = EdgeTTSService()
            if service.available:
                return service
        
        # Try all in order: ChatterBox -> VibeVoice -> Edge
        tts_classes = []
        try:
            from app3_video_assembler.services.chatterbox_service import ChatterboxTTSService
            tts_classes.append(ChatterboxTTSService)
        except ImportError:
            pass
        tts_classes.extend([VibeVoiceTTSService, EdgeTTSService])
        
        for ServiceClass in tts_classes:
            try:
                service = ServiceClass()
                if service.available:
                    return service
            except:
                continue
        
        raise RuntimeError(
            "No TTS available. Install ChatterBox: pip install chatterbox-tts\n"
            "Or Edge-TTS: pip install edge-tts"
        )

