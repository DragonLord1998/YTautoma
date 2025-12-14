"""
FFmpeg Service
Video composition and assembly using FFmpeg.
"""

import subprocess
import shutil
from pathlib import Path
from typing import List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import VIDEO_FPS, VIDEO_WIDTH, VIDEO_HEIGHT


class FFmpegService:
    """Video composition using FFmpeg"""
    
    def __init__(self):
        self.available = self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is installed"""
        return shutil.which("ffmpeg") is not None
    
    def _run_ffmpeg(self, args: List[str], description: str = "Processing") -> bool:
        """Run FFmpeg command"""
        cmd = ["ffmpeg", "-y"] + args  # -y to overwrite
        
        print(f"🔧 {description}...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300
            )
            
            if result.returncode != 0:
                error = result.stderr.decode()
                print(f"❌ FFmpeg error: {error[-500:]}")  # Last 500 chars
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            print("❌ FFmpeg timed out")
            return False
    
    def combine_video_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        audio_fade_out: float = 0.5
    ) -> Path:
        """
        Combine video with audio track.
        
        Args:
            video_path: Input video file
            audio_path: Input audio file
            output_path: Output combined file
            audio_fade_out: Fade out duration at end
            
        Returns:
            Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        args = [
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",  # End when shortest stream ends
            str(output_path)
        ]
        
        if self._run_ffmpeg(args, f"Combining {video_path.name} + {audio_path.name}"):
            return output_path
        else:
            raise RuntimeError("Failed to combine video and audio")
    
    def image_to_video(
        self,
        image_path: Path,
        output_path: Path,
        duration: float,
        zoom_effect: bool = True
    ) -> Path:
        """
        Convert static image to video with optional zoom effect.
        
        Args:
            image_path: Input image
            output_path: Output video
            duration: Video duration in seconds
            zoom_effect: Apply slow zoom (Ken Burns effect)
            
        Returns:
            Path to output video
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if zoom_effect:
            # Slow zoom in effect
            filter_complex = (
                f"scale=8000:-1,"
                f"zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={int(duration * VIDEO_FPS)}:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS}"
            )
        else:
            filter_complex = f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
        
        args = [
            "-loop", "1",
            "-i", str(image_path),
            "-vf", filter_complex,
            "-c:v", "libx264",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            "-r", str(VIDEO_FPS),
            str(output_path)
        ]
        
        if self._run_ffmpeg(args, f"Converting {image_path.name} to video"):
            return output_path
        else:
            raise RuntimeError("Failed to convert image to video")
    
    def concatenate_videos(
        self,
        video_paths: List[Path],
        output_path: Path
    ) -> Path:
        """
        Concatenate multiple videos into one.
        
        Args:
            video_paths: List of video files to concatenate
            output_path: Output concatenated video
            
        Returns:
            Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create concat file
        concat_file = output_path.parent / "concat_list.txt"
        with open(concat_file, "w") as f:
            for vp in video_paths:
                f.write(f"file '{vp.absolute()}'\n")
        
        args = [
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output_path)
        ]
        
        try:
            if self._run_ffmpeg(args, f"Concatenating {len(video_paths)} videos"):
                return output_path
            else:
                raise RuntimeError("Failed to concatenate videos")
        finally:
            concat_file.unlink(missing_ok=True)
    
    def add_background_music(
        self,
        video_path: Path,
        music_path: Path,
        output_path: Path,
        music_volume: float = 0.15,
        fade_out: float = 2.0
    ) -> Path:
        """
        Add background music to video, ducked under existing audio.
        
        Args:
            video_path: Input video with narration
            music_path: Background music file
            output_path: Output video
            music_volume: Volume level for music (0.0-1.0)
            fade_out: Fade out duration at end
            
        Returns:
            Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        filter_complex = (
            f"[1:a]volume={music_volume},afade=t=out:st=-{fade_out}:d={fade_out}[bgm];"
            f"[0:a][bgm]amix=inputs=2:duration=first[aout]"
        )
        
        args = [
            "-i", str(video_path),
            "-i", str(music_path),
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            str(output_path)
        ]
        
        if self._run_ffmpeg(args, "Adding background music"):
            return output_path
        else:
            raise RuntimeError("Failed to add background music")
    
    def add_audio_to_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path
    ) -> Path:
        """Add audio track to video (replacing any existing audio)"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        args = [
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path)
        ]
        
        if self._run_ffmpeg(args, "Adding audio to video"):
            return output_path
        else:
            raise RuntimeError("Failed to add audio")
    
    def get_duration(self, file_path: Path) -> float:
        """Get duration of video/audio file"""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
