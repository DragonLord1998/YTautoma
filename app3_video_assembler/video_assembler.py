#!/usr/bin/env python3
"""
App 3: Video Assembler (Local Models)
Generates voiceovers with VibeVoice and assembles final YouTube Short.
"""

import json
import argparse
import sys
import shutil
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OUTPUT_DIR, ASSETS_DIR
from shared.models import Story, AudioAsset, FinalVideo
from app3_video_assembler.services import TTSFactory, VibeVoiceTTSService, FFmpegService


class VideoAssembler:
    """Assemble final video from scenes, audio, and visuals"""
    
    def __init__(self, tts_engine: str = "vibevoice"):
        """
        Initialize video assembler.
        
        Args:
            tts_engine: 'vibevoice' or 'piper'
        """
        self.tts = TTSFactory.create(preferred=tts_engine)
        self.ffmpeg = FFmpegService()
        
        if not self.ffmpeg.available:
            raise RuntimeError("FFmpeg not found. Install FFmpeg to continue.")
    
    def assemble(
        self,
        story: Story,
        visuals_dir: Path,
        output_path: Path,
        add_bgm: bool = True,
        bgm_path: Optional[Path] = None
    ) -> FinalVideo:
        """
        Assemble final video from all components.
        
        Args:
            story: Story with scenes and narration
            visuals_dir: Directory containing visual assets
            output_path: Final output video path
            add_bgm: Whether to add background music
            bgm_path: Custom background music file
            
        Returns:
            FinalVideo metadata
        """
        visuals_dir = Path(visuals_dir)
        output_path = Path(output_path)
        
        temp_dir = output_path.parent / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎬 Assembling video: '{story.title}'")
        print("=" * 50)
        
        scene_videos = []
        
        for scene in story.scenes:
            print(f"\n📍 Scene {scene.scene_id}")
            
            scene_dir = visuals_dir / f"scene_{scene.scene_id:03d}"
            scene_temp = temp_dir / f"scene_{scene.scene_id:03d}"
            scene_temp.mkdir(exist_ok=True)
            
            # Step 1: Generate audio narration with VibeVoice
            print("   🔊 Generating narration with VibeVoice...")
            audio_path = scene_temp / "narration.wav"
            
            try:
                self.tts.synthesize(scene.narration, audio_path)
            except Exception as e:
                print(f"   ⚠️ TTS failed: {e}")
                continue
            
            # Step 2: Get video or create from image
            video_source = self._get_video_source(scene_dir)
            
            if video_source is None:
                print("   ⚠️ No video found, skipping scene")
                continue
            
            # Step 3: Combine video with audio
            print("   🔗 Combining video + audio...")
            scene_video = scene_temp / "scene_with_audio.mp4"
            
            self.ffmpeg.add_audio_to_video(
                video_path=video_source,
                audio_path=audio_path,
                output_path=scene_video
            )
            
            scene_videos.append(scene_video)
        
        if not scene_videos:
            raise RuntimeError("No scene videos to assemble")
        
        # Step 4: Concatenate all scenes
        print(f"\n🔗 Concatenating {len(scene_videos)} scenes...")
        combined_path = temp_dir / "combined.mp4"
        self.ffmpeg.concatenate_videos(scene_videos, combined_path)
        
        # Step 5: Add background music (optional)
        if add_bgm:
            bgm = bgm_path or self._get_default_bgm()
            if bgm and bgm.exists():
                print("🎵 Adding background music...")
                final_with_bgm = temp_dir / "final_with_bgm.mp4"
                self.ffmpeg.add_background_music(
                    combined_path,
                    bgm,
                    final_with_bgm,
                    music_volume=0.12
                )
                combined_path = final_with_bgm
            else:
                print("⚠️ No background music file found, skipping")
        
        # Step 6: Move to final output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(combined_path, output_path)
        
        # Get final duration
        duration = self.ffmpeg.get_duration(output_path)
        
        print(f"\n✅ Final video: {output_path}")
        print(f"   Duration: {duration:.1f}s")
        
        return FinalVideo(
            output_path=str(output_path),
            duration_seconds=duration,
            resolution="1080x1920",
            title=story.title
        )
    
    def _get_video_source(self, scene_dir: Path) -> Optional[Path]:
        """Get the best available video source for a scene"""
        # Priority: Wan 2.2 video > Image-based video
        
        video_clip = scene_dir / "video_clip.mp4"
        if video_clip.exists():
            return video_clip
        
        # Fall back to image -> video conversion
        for img_name in ["consistent_image.png", "base_image.png"]:
            img_path = scene_dir / img_name
            if img_path.exists():
                print(f"   🖼️ Converting {img_name} to video...")
                video_path = scene_dir / "image_video.mp4"
                self.ffmpeg.image_to_video(
                    img_path,
                    video_path,
                    duration=10,
                    zoom_effect=True
                )
                return video_path
        
        return None
    
    def _get_default_bgm(self) -> Optional[Path]:
        """Get default background music if available"""
        bgm_dir = ASSETS_DIR / "bgm"
        if bgm_dir.exists():
            for ext in [".mp3", ".wav", ".m4a"]:
                for bgm_file in bgm_dir.glob(f"*{ext}"):
                    return bgm_file
        return None
    
    def cleanup_temp(self, output_path: Path):
        """Remove temporary files"""
        temp_dir = output_path.parent / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("🧹 Cleaned up temp files")


def main():
    parser = argparse.ArgumentParser(description="Assemble final video with VibeVoice TTS")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input story JSON file")
    parser.add_argument("--visuals", "-v", type=str, required=True, help="Visuals directory")
    parser.add_argument("--output", "-o", type=str, help="Output video path")
    parser.add_argument("--tts", type=str, default="vibevoice", help="TTS engine (vibevoice/piper)")
    parser.add_argument("--no-bgm", action="store_true", help="Skip background music")
    parser.add_argument("--bgm", type=str, help="Custom background music file")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    
    args = parser.parse_args()
    
    story_path = Path(args.input)
    if not story_path.exists():
        print(f"❌ Story file not found: {story_path}")
        sys.exit(1)
    
    story = Story.model_validate_json(story_path.read_text())
    print(f"📖 Loaded story: '{story.title}'")
    
    visuals_dir = Path(args.visuals)
    if not visuals_dir.exists():
        print(f"❌ Visuals directory not found: {visuals_dir}")
        sys.exit(1)
    
    if args.output:
        output_path = Path(args.output)
    else:
        safe_title = "".join(c if c.isalnum() else "_" for c in story.title)
        output_path = OUTPUT_DIR / f"{safe_title}.mp4"
    
    try:
        assembler = VideoAssembler(tts_engine=args.tts)
        
        result = assembler.assemble(
            story=story,
            visuals_dir=visuals_dir,
            output_path=output_path,
            add_bgm=not args.no_bgm,
            bgm_path=Path(args.bgm) if args.bgm else None
        )
        
        if not args.keep_temp:
            assembler.cleanup_temp(output_path)
        
        print("\n📋 Summary:")
        print(f"   Title: {result.title}")
        print(f"   Duration: {result.duration_seconds:.1f}s")
        print(f"   Output: {result.output_path}")
        
        return result.output_path
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
