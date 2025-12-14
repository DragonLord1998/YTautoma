#!/usr/bin/env python3
"""
App 2: Visual Generator (Local Models)
Generates consistent images and converts them to video clips.
Pipeline: Z-Image (diffusers) → Qwen-Image-Edit → Wan 2.2 (local)
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OUTPUT_DIR, LOW_VRAM_MODE
from shared.models import Story, VisualAsset
from services import ZImageService, QwenImageEditService, SimpleConsistencyService, Wan22VideoService


class VisualGenerator:
    """Generate visual assets for story scenes using local models"""
    
    def __init__(
        self,
        enable_consistency: bool = True,
        enable_video: bool = True,
        low_vram: bool = LOW_VRAM_MODE
    ):
        """
        Initialize visual generator with local models.
        
        Args:
            enable_consistency: Use Qwen-Image-Edit for character consistency
            enable_video: Generate video clips with Wan 2.2
            low_vram: Enable memory-saving mode
        """
        self.zimage = ZImageService()
        
        # Try Qwen-Image-Edit, fall back to simple service
        if enable_consistency:
            try:
                self.consistency = QwenImageEditService()
            except Exception:
                print("⚠️ Using fallback consistency service")
                self.consistency = SimpleConsistencyService()
        else:
            self.consistency = None
        
        self.video = Wan22VideoService() if enable_video else None
        
        self.enable_consistency = enable_consistency
        self.enable_video = enable_video
        self.low_vram = low_vram
        
        self.character_reference_path: Optional[Path] = None
    
    def process_story(self, story: Story, output_dir: Path) -> list[VisualAsset]:
        """
        Process all scenes in a story.
        
        Args:
            story: Story object with scenes
            output_dir: Directory to save visual assets
            
        Returns:
            List of VisualAsset objects
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        assets = []
        
        print(f"\n🎬 Processing {len(story.scenes)} scenes for '{story.title}'")
        print("=" * 50)
        
        # Pre-load Z-Image model
        print("\n🔄 Loading Z-Image model...")
        self.zimage.load_model()
        
        for i, scene in enumerate(story.scenes):
            print(f"\n📍 Scene {scene.scene_id}/{len(story.scenes)}")
            
            scene_dir = output_dir / f"scene_{scene.scene_id:03d}"
            scene_dir.mkdir(exist_ok=True)
            
            try:
                asset = self._process_scene(
                    scene=scene,
                    scene_dir=scene_dir,
                    character_reference=story.character_reference,
                    is_first_scene=(i == 0)
                )
                assets.append(asset)
                
            except Exception as e:
                print(f"❌ Error processing scene {scene.scene_id}: {e}")
                assets.append(VisualAsset(
                    scene_id=scene.scene_id,
                    base_image_path=str(scene_dir / "base_image.png")
                ))
        
        # Unload Z-Image to free memory for video generation
        if self.low_vram and self.enable_video:
            self.zimage.unload_model()
        
        print(f"\n✅ Processed {len(assets)} scenes")
        return assets
    
    def _process_scene(
        self,
        scene,
        scene_dir: Path,
        character_reference: Optional[str],
        is_first_scene: bool
    ) -> VisualAsset:
        """Process a single scene through the visual pipeline"""
        
        # Step 1: Generate base image with Z-Image
        print(f"   🎨 Generating base image...")
        base_image_path = scene_dir / "base_image.png"
        
        enhanced_prompt = self._enhance_prompt(
            scene.visual_prompt,
            character_reference,
            scene.character_description
        )
        
        self.zimage.generate_to_file(
            prompt=enhanced_prompt,
            output_path=base_image_path
        )
        
        # Step 2: Save first scene as character reference
        if is_first_scene and self.enable_consistency:
            self.character_reference_path = scene_dir / "character_reference.png"
            self.character_reference_path.write_bytes(base_image_path.read_bytes())
            print(f"   📌 Saved character reference")
        
        # Step 3: Apply character consistency (if not first scene)
        consistent_image_path = None
        if self.enable_consistency and self.consistency and not is_first_scene:
            if self.character_reference_path and self.character_reference_path.exists():
                print(f"   🔄 Applying character consistency...")
                consistent_image_path = scene_dir / "consistent_image.png"
                
                try:
                    self.consistency.apply_to_file(
                        source_image=base_image_path,
                        reference_image=self.character_reference_path,
                        output_path=consistent_image_path,
                        prompt=scene.visual_prompt,
                        character_description=scene.character_description or ""
                    )
                except Exception as e:
                    print(f"   ⚠️ Consistency failed: {e}")
                    consistent_image_path = None
        
        final_image = consistent_image_path or base_image_path
        
        # Step 4: Generate video clip with Wan 2.2
        video_path = None
        if self.enable_video and self.video and self.video.available:
            print(f"   🎬 Generating video clip with Wan 2.2...")
            video_path = scene_dir / "video_clip.mp4"
            
            motion_prompt = self._build_motion_prompt(scene)
            
            try:
                self.video.generate_to_file(
                    image_path=final_image,
                    output_path=video_path,
                    prompt=motion_prompt
                )
            except Exception as e:
                print(f"   ⚠️ Video generation failed: {e}")
                video_path = None
        
        return VisualAsset(
            scene_id=scene.scene_id,
            base_image_path=str(base_image_path),
            consistent_image_path=str(consistent_image_path) if consistent_image_path else None,
            video_clip_path=str(video_path) if video_path else None
        )
    
    def _enhance_prompt(
        self,
        visual_prompt: str,
        character_reference: Optional[str],
        character_description: Optional[str]
    ) -> str:
        """Enhance prompt for better Z-Image results"""
        parts = [visual_prompt]
        
        if character_reference:
            parts.append(f"Main character: {character_reference}")
        
        if character_description:
            parts.append(f"Character in scene: {character_description}")
        
        parts.append("vertical composition, 9:16 aspect ratio, cinematic, photorealistic, high detail, 8K")
        
        return ". ".join(parts)
    
    def _build_motion_prompt(self, scene) -> str:
        """Build motion prompt for video generation"""
        prompt_lower = scene.visual_prompt.lower()
        
        camera_hints = []
        if "close-up" in prompt_lower or "closeup" in prompt_lower:
            camera_hints.append("subtle zoom out")
        elif "wide shot" in prompt_lower:
            camera_hints.append("slow pan")
        elif "dramatic" in prompt_lower:
            camera_hints.append("cinematic camera movement")
        else:
            camera_hints.append("gentle motion, subtle movement")
        
        return f"{', '.join(camera_hints)}, {scene.visual_prompt}"
    
    def save_assets_manifest(self, assets: list[VisualAsset], output_path: Path):
        """Save manifest of generated assets"""
        manifest = {
            "assets": [a.model_dump() for a in assets],
            "total_scenes": len(assets),
            "character_reference": str(self.character_reference_path) if self.character_reference_path else None
        }
        
        output_path.write_text(json.dumps(manifest, indent=2))
        print(f"💾 Manifest saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate visual assets for story (local models)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input story JSON file")
    parser.add_argument("--output", "-o", type=str, help="Output directory for visuals")
    parser.add_argument("--no-consistency", action="store_true", help="Disable character consistency")
    parser.add_argument("--no-video", action="store_true", help="Generate images only, no video")
    parser.add_argument("--images-only", action="store_true", help="Alias for --no-video")
    parser.add_argument("--low-vram", action="store_true", default=True, help="Enable memory-saving mode")
    
    args = parser.parse_args()
    
    story_path = Path(args.input)
    if not story_path.exists():
        print(f"❌ Story file not found: {story_path}")
        sys.exit(1)
    
    story = Story.model_validate_json(story_path.read_text())
    print(f"📖 Loaded story: '{story.title}'")
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = OUTPUT_DIR / "visuals" / story_path.stem
    
    generator = VisualGenerator(
        enable_consistency=not args.no_consistency,
        enable_video=not (args.no_video or args.images_only),
        low_vram=args.low_vram
    )
    
    try:
        assets = generator.process_story(story, output_dir)
        
        generator.save_assets_manifest(
            assets,
            output_dir / "assets_manifest.json"
        )
        
        print("\n📋 Summary:")
        print(f"   Output: {output_dir}")
        print(f"   Scenes: {len(assets)}")
        print(f"   Images: {sum(1 for a in assets if a.base_image_path)}")
        print(f"   Videos: {sum(1 for a in assets if a.video_clip_path)}")
        
        return str(output_dir)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
