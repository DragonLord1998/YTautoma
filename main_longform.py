#!/usr/bin/env python3
"""
Long-Form YouTube Video Automation
Generates 20-minute videos (20 x 1-minute parts) using local AI models.

Pipeline:
1. Research Agent → Trending topic discovery
2. Story Generator → 20-part narrative with cliffhangers
3. Visual Generator → Z-Image + Wan 2.2 14B (720p)
4. Video Assembler → ChatterBox TTS + FFmpeg
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from shared.config import OUTPUT_DIR, TOTAL_PARTS
from app0_research_agent.research_agent import ResearchAgent
from app1_story_generator.longform_story_generator import LongFormStoryGenerator
from app2_visual_generator.visual_generator import VisualGenerator
from app3_video_assembler.video_assembler import VideoAssembler


def run_longform_pipeline(
    idea: str,
    skip_research: bool = False,
    skip_visuals: bool = False,
    skip_video: bool = False,
    images_only: bool = False,
    output_dir: Path = None,
    start_part: int = 1,
    end_part: int = None
):
    """
    Run the complete long-form video generation pipeline.
    
    Args:
        idea: User's content idea
        skip_research: Skip research phase
        skip_visuals: Skip visual generation
        skip_video: Skip final video assembly
        images_only: Generate images only (no Wan 2.2 video)
        output_dir: Custom output directory
        start_part: Start from this part (for resuming)
        end_part: End at this part (for partial generation)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or OUTPUT_DIR / f"longform_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    end_part = end_part or TOTAL_PARTS
    
    print("\n" + "=" * 60)
    print("🎬 LONG-FORM VIDEO GENERATION PIPELINE")
    print(f"   Creating {TOTAL_PARTS} x 1-minute parts (20 minutes total)")
    print("=" * 60)
    print(f"\n💡 Idea: {idea}")
    
    # =========================================
    # STEP 1: Research (Optional)
    # =========================================
    if not skip_research:
        print("\n🔬 STEP 1: Research & Topic Discovery")
        print("-" * 40)
        
        research_agent = ResearchAgent()
        research = research_agent.research(idea)
        research_agent.save_research(research, output_dir / "research.json")
        
        print(f"   ✅ Refined Topic: {research.refined_topic}")
        print(f"   📊 Themes: {', '.join(research.key_themes[:3])}")
    else:
        print("\n⏭️ STEP 1: Skipped (research)")
    
    # =========================================
    # STEP 2: Generate Story
    # =========================================
    print("\n📖 STEP 2: Story Generation (20 Parts)")
    print("-" * 40)
    
    story_generator = LongFormStoryGenerator(use_research=not skip_research)
    story = story_generator.generate(idea, use_research=not skip_research)
    story_path = story_generator.save_story(story, output_dir / "story.json")
    
    print(f"   ✅ Story: '{story.title}'")
    print(f"   📄 Parts: {len(story.parts) if story.is_longform else 'N/A'}")
    print(f"   🎬 Scenes: {len(story.all_scenes)}")
    
    if skip_visuals and skip_video:
        print("\n✅ Pipeline complete (story only)")
        return str(story_path)
    
    # =========================================
    # STEP 3: Generate Visuals (per part)
    # =========================================
    if not skip_visuals:
        print("\n🎨 STEP 3: Visual Generation")
        print("-" * 40)
        
        visual_generator = VisualGenerator(enable_video=not images_only)
        
        if story.is_longform:
            # Process each part separately
            for part in story.parts:
                if part.part_id < start_part or part.part_id > end_part:
                    continue
                    
                print(f"\n   📍 Part {part.part_id}/{len(story.parts)}: {part.part_title}")
                
                part_dir = output_dir / f"part_{part.part_id:02d}"
                
                # Create a mini-story for this part
                from shared.models import Story
                part_story = Story(
                    title=f"{story.title} - {part.part_title}",
                    topic=story.topic,
                    scenes=part.scenes,
                    character_reference=story.character_reference,
                    total_duration=60
                )
                
                assets = visual_generator.process_story(part_story, part_dir / "visuals")
                visual_generator.save_assets_manifest(assets, part_dir / "manifest.json")
        else:
            # Fallback for short-form
            visuals_dir = output_dir / "visuals"
            assets = visual_generator.process_story(story, visuals_dir)
            visual_generator.save_assets_manifest(assets, visuals_dir / "manifest.json")
    else:
        print("\n⏭️ STEP 3: Skipped (using existing visuals)")
    
    if skip_video:
        print("\n✅ Pipeline complete (story + visuals)")
        return str(output_dir)
    
    # =========================================
    # STEP 4: Assemble Videos (per part)
    # =========================================
    print("\n🎬 STEP 4: Video Assembly")
    print("-" * 40)
    
    assembler = VideoAssembler()
    part_videos = []
    
    if story.is_longform:
        for part in story.parts:
            if part.part_id < start_part or part.part_id > end_part:
                continue
                
            print(f"\n   📍 Assembling Part {part.part_id}/{len(story.parts)}")
            
            part_dir = output_dir / f"part_{part.part_id:02d}"
            visuals_dir = part_dir / "visuals"
            
            # Create mini-story for assembly
            from shared.models import Story
            part_story = Story(
                title=f"{story.title} - {part.part_title}",
                topic=story.topic,
                scenes=part.scenes,
                character_reference=story.character_reference,
                total_duration=60
            )
            
            part_video = part_dir / f"part_{part.part_id:02d}.mp4"
            
            try:
                result = assembler.assemble(
                    story=part_story,
                    visuals_dir=visuals_dir,
                    output_path=part_video,
                    add_bgm=False  # Add BGM at the end
                )
                part_videos.append(part_video)
                print(f"   ✅ Part {part.part_id}: {result.duration_seconds:.1f}s")
            except Exception as e:
                print(f"   ❌ Part {part.part_id} failed: {e}")
    
    # =========================================
    # STEP 5: Concatenate All Parts
    # =========================================
    if len(part_videos) > 1:
        print("\n🔗 STEP 5: Concatenating Parts")
        print("-" * 40)
        
        from app3_video_assembler.services import FFmpegService
        ffmpeg = FFmpegService()
        
        safe_title = "".join(c if c.isalnum() else "_" for c in story.title)[:50]
        final_video = output_dir / f"{safe_title}_FULL.mp4"
        
        ffmpeg.concatenate_videos(part_videos, final_video)
        
        # Add background music to final video
        bgm_video = output_dir / f"{safe_title}_FINAL.mp4"
        assembler._get_default_bgm()  # Check for BGM
        
        final_duration = ffmpeg.get_duration(final_video)
        
        print(f"\n   ✅ Final video: {final_video.name}")
        print(f"   ⏱️ Duration: {final_duration // 60:.0f}m {final_duration % 60:.0f}s")
    
    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 60)
    print("🎉 LONG-FORM PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"   📁 Project: {output_dir}")
    print(f"   📖 Story: {story.title}")
    print(f"   🎬 Parts: {len(part_videos)}")
    if part_videos:
        print(f"   📺 Output: {part_videos[-1].parent.parent.name}/")
    print("\n   Upload to YouTube as a long-form video! 🚀")
    
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate 20-minute YouTube videos using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_longform.py -i "Ancient mysteries of Egypt"
  python main_longform.py -i "The rise of AI" --images-only
  python main_longform.py -i "Space exploration" --parts 1-5
        """
    )
    
    parser.add_argument(
        "--idea", "-i",
        type=str,
        required=True,
        help="Your content idea"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory"
    )
    parser.add_argument(
        "--no-research",
        action="store_true",
        help="Skip research phase"
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip visual generation"
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip video assembly"
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Generate images only (no Wan 2.2 video)"
    )
    parser.add_argument(
        "--story-only",
        action="store_true",
        help="Generate story only"
    )
    parser.add_argument(
        "--parts",
        type=str,
        help="Part range to generate (e.g., '1-5' or '10-20')"
    )
    
    args = parser.parse_args()
    
    # Parse part range
    start_part, end_part = 1, TOTAL_PARTS
    if args.parts:
        try:
            parts = args.parts.split("-")
            start_part = int(parts[0])
            end_part = int(parts[1]) if len(parts) > 1 else start_part
        except:
            print(f"⚠️ Invalid parts format: {args.parts}, using all parts")
    
    # Handle story-only flag
    if args.story_only:
        args.skip_visuals = True
        args.skip_video = True
    
    try:
        result = run_longform_pipeline(
            idea=args.idea,
            skip_research=args.no_research,
            skip_visuals=args.skip_visuals,
            skip_video=args.skip_video,
            images_only=args.images_only,
            output_dir=Path(args.output) if args.output else None,
            start_part=start_part,
            end_part=end_part
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
