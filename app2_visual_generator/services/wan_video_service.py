"""
Wan 2.2 Video Service (Local)
Image-to-Video generation using Wan 2.2 I2V model.
https://github.com/Wan-Video/Wan2.2
"""

import subprocess
import os
from pathlib import Path
from typing import Optional
import sys
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import (
    WAN_REPO_PATH, WAN_MODEL_PATH, WAN_VIDEO_SIZE, 
    WAN_T5_CPU, WAN_OFFLOAD_MODEL, VIDEO_FPS
)


class Wan22VideoService:
    """Generate video clips from images using local Wan 2.2 I2V model"""
    
    def __init__(
        self,
        repo_path: str = WAN_REPO_PATH,
        model_path: str = WAN_MODEL_PATH
    ):
        """
        Initialize Wan 2.2 video service.
        
        Args:
            repo_path: Path to cloned Wan2.2 repository
            model_path: Path to downloaded model weights
        """
        self.repo_path = Path(repo_path)
        self.model_path = Path(model_path)
        self._check_installation()
    
    def _check_installation(self):
        """Check if Wan 2.2 is properly installed"""
        self.available = False
        
        if not self.repo_path.exists():
            print(f"⚠️ Wan 2.2 repo not found at {self.repo_path}")
            print("   Clone with: git clone https://github.com/Wan-Video/Wan2.2.git")
            return
        
        generate_script = self.repo_path / "generate.py"
        if not generate_script.exists():
            print(f"⚠️ generate.py not found in {self.repo_path}")
            return
        
        if not self.model_path.exists():
            print(f"⚠️ Model not found at {self.model_path}")
            print("   Download with: huggingface-cli download Wan-AI/Wan2.2-I2V-A14B")
            return
        
        self.available = True
        print(f"✅ Wan 2.2 ready at {self.repo_path}")
    
    def generate_video(
        self,
        image_path: Path,
        prompt: str,
        output_path: Optional[Path] = None,
        size: str = WAN_VIDEO_SIZE,
        offload_model: bool = WAN_OFFLOAD_MODEL,
        t5_cpu: bool = WAN_T5_CPU,
        sample_steps: int = 20  # Lower = faster, default 50 is slow
    ) -> Path:
        """
        Generate video from image using Wan 2.2 I2V.
        
        Args:
            image_path: Path to input image
            prompt: Motion/scene description
            output_path: Full output file path (optional)
            size: Video size (e.g., '720*1280' for vertical)
            offload_model: Offload model to save VRAM
            t5_cpu: Run T5 encoder on CPU
            sample_steps: Number of inference steps (lower = faster, 20-30 recommended)
            
        Returns:
            Path to generated video
        """
        if not self.available:
            raise RuntimeError(
                "Wan 2.2 not available. Please install:\n"
                "1. git clone https://github.com/Wan-Video/Wan2.2.git\n"
                "2. huggingface-cli download Wan-AI/Wan2.2-I2V-A14B"
            )
        
        image_path = Path(image_path).absolute()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Build command
        cmd = [
            sys.executable,
            str(self.repo_path / "generate.py"),
            "--task", "ti2v-5B",  # Matches Wan2.2-TI2V-5B model
            "--size", size,
            "--ckpt_dir", str(self.model_path),
            "--image", str(image_path),
            "--prompt", prompt,
        ]
        
        if offload_model:
            cmd.extend(["--offload_model", "True"])
        
        if t5_cpu:
            cmd.append("--t5_cpu")
        
        # Add dtype conversion for memory savings
        cmd.append("--convert_model_dtype")
        
        # Add sample steps for speed control
        cmd.extend(["--sample_steps", str(sample_steps)])
        
        # Use --save_file for output path (not --save_dir)
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--save_file", str(output_path)])
        
        print(f"🎬 Generating video with Wan 2.2...")
        print(f"   Image: {image_path.name}")
        print(f"   Prompt: {prompt[:60]}...")
        
        # Run generation
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode != 0:
                print(f"❌ Wan 2.2 error:\n{result.stderr[-1000:]}")
                raise RuntimeError(f"Video generation failed: {result.stderr[-500:]}")
            
            # If output_path was specified, return it
            if output_path and output_path.exists():
                print(f"✅ Video generated: {output_path}")
                return output_path
            
            # Otherwise find generated video in default output
            save_dir = self.repo_path / "output"
            video_files = list(save_dir.glob("*.mp4"))
            
            if not video_files:
                raise RuntimeError("No video file generated")
            
            # Get most recent video
            latest_video = max(video_files, key=lambda p: p.stat().st_mtime)
            
            print(f"✅ Video generated: {latest_video}")
            return latest_video
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Video generation timed out (30 min limit)")
    
    def generate_to_file(
        self,
        image_path: Path,
        output_path: Path,
        prompt: str,
        **kwargs
    ) -> Path:
        """Generate video directly to specific output path"""
        
        output_path = Path(output_path)
        return self.generate_video(
            image_path=image_path,
            prompt=prompt,
            output_path=output_path,
            **kwargs
        )


class Wan22TI2VService(Wan22VideoService):
    """
    Text-Image-to-Video using Wan2.2-TI2V-5B (lighter model).
    Runs on RTX 4090 with 24GB VRAM.
    """
    
    def __init__(self, repo_path: str = WAN_REPO_PATH):
        self.repo_path = Path(repo_path)
        self.model_path = Path(repo_path).parent / "Wan2.2-TI2V-5B"
        self._check_installation()
    
    def generate_video(
        self,
        image_path: Path,
        prompt: str,
        output_path: Optional[Path] = None,
        size: str = "704*1280",  # TI2V uses different resolution
        sample_steps: int = 20,  # Lower = faster
        **kwargs
    ) -> Path:
        """Generate using TI2V-5B task"""
        
        if not self.available:
            raise RuntimeError("Wan 2.2 TI2V-5B not available")
        
        image_path = Path(image_path).absolute()
        
        cmd = [
            sys.executable,
            str(self.repo_path / "generate.py"),
            "--task", "ti2v-5B",
            "--size", size,
            "--ckpt_dir", str(self.model_path),
            "--image", str(image_path),
            "--prompt", prompt,
            "--offload_model", "True",
            "--convert_model_dtype",
            "--t5_cpu",
            "--sample_steps", str(sample_steps),
        ]
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--save_file", str(output_path)])
        
        print(f"🎬 Generating video with Wan 2.2 TI2V-5B...")
        
        result = subprocess.run(
            cmd,
            cwd=str(self.repo_path),
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Video generation failed: {result.stderr[-500:]}")
        
        if output_path and output_path.exists():
            return output_path
        
        save_dir = self.repo_path / "output"
        video_files = list(save_dir.glob("*.mp4"))
        
        if not video_files:
            raise RuntimeError("No video file generated")
        
        return max(video_files, key=lambda p: p.stat().st_mtime)
