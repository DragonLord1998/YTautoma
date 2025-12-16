#!/usr/bin/env python3
"""
Long-Form Story Generator
Generates 20-part narratives (20 x 1-minute) using research input and local LLM.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from shared.config import (
    OLLAMA_MODEL, OLLAMA_BASE_URL, OUTPUT_DIR, 
    TOTAL_PARTS, SCENES_PER_PART, PART_DURATION,
    RESEARCH_MODEL
)
from shared.models import Story, Part, Scene
from app0_research_agent.research_agent import ResearchAgent, ResearchResult


class LongFormStoryGenerator:
    """Generate 20-part long-form stories using local LLM"""
    
    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        use_research: bool = True
    ):
        self.model = model
        self.base_url = base_url
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.research_agent = ResearchAgent(model=RESEARCH_MODEL) if use_research else None
    
    def _load_system_prompt(self) -> str:
        """Load the long-form story generation system prompt"""
        prompt_path = self.prompts_dir / "longform_story_system.txt"
        if prompt_path.exists():
            return prompt_path.read_text()
        # Fallback to short-form prompt
        return (self.prompts_dir / "story_system.txt").read_text()
    
    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama API with the given prompts"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "top_p": 0.9,
                "num_predict": 16000  # Long-form needs more tokens
            }
        }
        
        print(f"🤖 Generating story with {self.model}...")
        print("   (This may take several minutes for 20 parts)")
        
        try:
            response = requests.post(url, json=payload, timeout=600)  # 10 min timeout
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: 'ollama serve'"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("Story generation timed out.")
    
    def _parse_story_json(self, raw_response: str) -> Story:
        """Parse LLM response into Story model"""
        cleaned = raw_response.strip()
        
        # Remove markdown code blocks
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            cleaned = "\n".join(lines)
        
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error: {e}")
            print(f"Raw response (first 1000 chars):\n{raw_response[:1000]}...")
            raise ValueError("Failed to parse story JSON from LLM response")
        
        # Convert to Story model
        story = Story(**data)
        
        # Validate part count
        if story.is_longform and len(story.parts) != TOTAL_PARTS:
            print(f"⚠️ Expected {TOTAL_PARTS} parts, got {len(story.parts)}")
        
        return story
    
    def generate(
        self,
        idea: str,
        use_research: bool = True
    ) -> Story:
        """
        Generate a complete 20-part story.
        
        Args:
            idea: User's content idea
            use_research: Whether to use research agent first
            
        Returns:
            Story with 20 parts
        """
        print(f"\n🎬 Generating Long-Form Video Story")
        print(f"   Idea: {idea}")
        print("=" * 50)
        
        # Step 1: Research (optional)
        research_context = ""
        if use_research and self.research_agent:
            try:
                research = self.research_agent.research(idea)
                research_context = self._format_research(research)
                print(f"   ✅ Research complete: {research.refined_topic}")
            except Exception as e:
                print(f"   ⚠️ Research failed: {e}")
        
        # Step 2: Generate story
        system_prompt = self._load_system_prompt()
        
        user_prompt = f"""Create an epic 20-minute video divided into {TOTAL_PARTS} parts, each ~{PART_DURATION} seconds.

**User's Idea:** {idea}

{research_context}

**Structure Requirements:**
- {TOTAL_PARTS} parts, each with {SCENES_PER_PART} scenes
- Each scene ~{PART_DURATION // SCENES_PER_PART} seconds
- Total runtime: ~{TOTAL_PARTS * PART_DURATION // 60} minutes
- End each part with a cliffhanger

Return ONLY valid JSON, no other text."""
        
        raw_response = self._call_ollama(system_prompt, user_prompt)
        
        # Parse and validate
        story = self._parse_story_json(raw_response)
        
        print(f"\n✅ Generated: '{story.title}'")
        print(f"   Parts: {len(story.parts) if story.is_longform else 'N/A'}")
        print(f"   Total scenes: {len(story.all_scenes)}")
        
        return story
    
    def _format_research(self, research: ResearchResult) -> str:
        """Format research results for story prompt"""
        return f"""
**Research Insights:**
- Refined Topic: {research.refined_topic}
- Key Themes: {', '.join(research.key_themes[:5])}
- Target Audience: {research.target_audience}
- Emotional Journey: {research.emotional_journey}

**Trending Angles to Consider:**
{chr(10).join(f'- {angle}' for angle in research.trending_angles[:3])}

**Character Archetypes:**
{chr(10).join(f'- {char}' for char in research.character_archetypes[:3])}
"""
    
    def save_story(self, story: Story, output_path: Optional[Path] = None) -> Path:
        """Save story to JSON file"""
        if output_path is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"longform_story_{timestamp}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(story.model_dump_json(indent=2))
        
        print(f"💾 Saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate long-form video stories")
    parser.add_argument("--idea", "-i", type=str, required=True, help="Your content idea")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file path")
    parser.add_argument("--model", "-m", type=str, default=OLLAMA_MODEL, help=f"Ollama model")
    parser.add_argument("--no-research", action="store_true", help="Skip research step")
    
    args = parser.parse_args()
    
    generator = LongFormStoryGenerator(
        model=args.model,
        use_research=not args.no_research
    )
    
    try:
        story = generator.generate(
            idea=args.idea,
            use_research=not args.no_research
        )
        
        output_path = Path(args.output) if args.output else None
        saved_path = generator.save_story(story, output_path)
        
        print("\n📋 Story Summary:")
        print(f"   Title: {story.title}")
        print(f"   Parts: {len(story.parts) if story.parts else 0}")
        print(f"   Scenes: {len(story.all_scenes)}")
        print(f"   Duration: ~{story.total_duration // 60} minutes")
        
        return str(saved_path)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
