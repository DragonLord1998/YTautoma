#!/usr/bin/env python3
"""
YouTube Shorts Automation - Main Orchestrator
Runs the complete pipeline: Story → Visuals → Video
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from shared.config import OUTPUT_DIR
from app1_story_generator.story_generator import StoryGenerator
from app2_visual_generator.visual_generator import VisualGenerator
from app3_video_assembler.video_assembler import VideoAssembler


def run_pipeline(
    category: str = None,
    topic: str = None,
    skip_visuals: bool = False,
    skip_video: bool = False,
    images_only: bool = False,
    output_dir: Path = None
):
    """
    Run the complete YouTube Shorts generation pipeline.
    
    Args:
        category: Story category (mystery, horror, sci-fi, etc.)
        topic: Custom topic override
        skip_visuals: Skip visual generation (use existing)
        skip_video: Skip final video assembly
        images_only: Generate images only, no AI video clips
        output_dir: Custom output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or OUTPUT_DIR / f"project_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🎬 YOUTUBE SHORTS AUTOMATION PIPELINE")
    print("=" * 60)
    
    # =========================================
    # STEP 1: Generate Story
    # =========================================
    print("\n📖 STEP 1: Story Generation")
    print("-" * 40)
    
    story_generator = StoryGenerator()
    story = story_generator.generate(category=category, topic_override=topic)
    story_path = story_generator.save_story(story, output_dir / "story.json")
    
    print(f"   ✅ Story: '{story.title}'")
    print(f"   📄 File: {story_path}")
    
    if skip_visuals and skip_video:
        print("\n✅ Pipeline complete (story only)")
        return str(story_path)
    
    # =========================================
    # STEP 2: Generate Visuals
    # =========================================
    visuals_dir = output_dir / "visuals"
    
    if not skip_visuals:
        print("\n🎨 STEP 2: Visual Generation")
        print("-" * 40)
        
        visual_generator = VisualGenerator(
            enable_video=not images_only
        )
        assets = visual_generator.process_story(story, visuals_dir)
        visual_generator.save_assets_manifest(assets, visuals_dir / "manifest.json")
        
        print(f"   ✅ Generated {len(assets)} scenes")
        print(f"   📁 Directory: {visuals_dir}")
    else:
        print("\n⏭️ STEP 2: Skipped (using existing visuals)")
    
    if skip_video:
        print("\n✅ Pipeline complete (story + visuals)")
        return str(visuals_dir)
    
    # =========================================
    # STEP 3: Assemble Final Video
    # =========================================
    print("\n🎬 STEP 3: Video Assembly")
    print("-" * 40)
    
    safe_title = "".join(c if c.isalnum() else "_" for c in story.title)[:50]
    final_video_path = output_dir / f"{safe_title}.mp4"
    
    assembler = VideoAssembler()
    result = assembler.assemble(
        story=story,
        visuals_dir=visuals_dir,
        output_path=final_video_path
    )
    assembler.cleanup_temp(final_video_path)
    
    print(f"   ✅ Final video: {result.output_path}")
    print(f"   ⏱️ Duration: {result.duration_seconds:.1f}s")
    
    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"   📁 Project: {output_dir}")
    print(f"   📖 Story: {story.title}")
    print(f"   🎬 Video: {final_video_path.name}")
    print(f"   ⏱️ Duration: {result.duration_seconds:.1f}s")
    print("\n   Upload to YouTube Shorts and enjoy! 🚀")
    
    return str(final_video_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate YouTube Shorts automatically using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Random topic
  python main.py -c mystery                # Mystery category
  python main.py -t "A robot learns love"  # Custom topic
  python main.py --images-only             # Skip video generation (faster)
        """
    )
    
    parser.add_argument(
        "--category", "-c",
        type=str,
        choices=["mystery", "horror", "sci-fi", "fantasy", "inspirational", "thriller"],
        help="Story category"
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        help="Custom story topic/premise"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory"
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip visual generation (use existing)"
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip final video assembly"
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Generate images only, no AI video clips (faster)"
    )
    parser.add_argument(
        "--story-only",
        action="store_true",
        help="Generate story only (no visuals or video)"
    )
    
    args = parser.parse_args()
    
    # Handle story-only flag
    if args.story_only:
        args.skip_visuals = True
        args.skip_video = True
    
    try:
        result = run_pipeline(
            category=args.category,
            topic=args.topic,
            skip_visuals=args.skip_visuals,
            skip_video=args.skip_video,
            images_only=args.images_only,
            output_dir=Path(args.output) if args.output else None
        )
        print(f"\n📍 Output: {result}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
