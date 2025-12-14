#!/usr/bin/env python3
"""
App 1: Story Generator
Generates narrated story scripts using Qwen 3 4B via Ollama.
"""

import json
import random
import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from shared.config import OLLAMA_MODEL, OLLAMA_BASE_URL, OUTPUT_DIR, SCENES_COUNT, TARGET_DURATION
from shared.models import Story, Scene

# Default model is now gemma3:27b-abliterated


class StoryGenerator:
    """Generate stories using local LLM via Ollama"""
    
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.topics_file = Path(__file__).parent / "topics.json"
        
    def _load_system_prompt(self) -> str:
        """Load the story generation system prompt"""
        prompt_path = self.prompts_dir / "story_system.txt"
        return prompt_path.read_text()
    
    def _load_topics(self) -> dict:
        """Load topics from JSON file"""
        return json.loads(self.topics_file.read_text())
    
    def _get_random_topic(self, category: Optional[str] = None) -> tuple[str, str]:
        """Get a random topic, optionally from a specific category"""
        topics_data = self._load_topics()
        
        if category:
            # Find specific category
            for cat in topics_data["topics"]:
                if cat["category"].lower() == category.lower():
                    prompt = random.choice(cat["prompts"])
                    return category, prompt
            raise ValueError(f"Category '{category}' not found")
        else:
            # Random category and prompt
            cat = random.choice(topics_data["topics"])
            prompt = random.choice(cat["prompts"])
            return cat["category"], prompt
    
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
                "num_predict": 2000
            }
        }
        
        print(f"🤖 Calling Ollama ({self.model})...")
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: 'ollama serve'"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("Ollama request timed out. Try a smaller model or increase timeout.")
    
    def _parse_story_json(self, raw_response: str) -> Story:
        """Parse LLM response into Story model"""
        # Clean up response - remove markdown code blocks if present
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            # Remove code block markers
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            cleaned = "\n".join(lines)
        
        # Parse JSON
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error: {e}")
            print(f"Raw response:\n{raw_response[:500]}...")
            raise ValueError("Failed to parse story JSON from LLM response")
        
        # Validate with Pydantic
        story = Story(**data)
        
        # Validate scene count
        if len(story.scenes) != SCENES_COUNT:
            print(f"⚠️ Expected {SCENES_COUNT} scenes, got {len(story.scenes)}")
        
        return story
    
    def generate(self, category: Optional[str] = None, topic_override: Optional[str] = None) -> Story:
        """Generate a complete story"""
        # Get topic
        if topic_override:
            cat = category or "custom"
            topic = topic_override
        else:
            cat, topic = self._get_random_topic(category)
        
        print(f"📖 Topic: [{cat}] {topic}")
        
        # Build prompts
        system_prompt = self._load_system_prompt()
        user_prompt = f"""Create a compelling 60-second story based on this premise:

**Topic Category:** {cat}
**Premise:** {topic}

Generate exactly {SCENES_COUNT} scenes, each {TARGET_DURATION // SCENES_COUNT} seconds long.
Return ONLY valid JSON, no other text."""
        
        # Generate story
        raw_response = self._call_ollama(system_prompt, user_prompt)
        
        # Parse and validate
        story = self._parse_story_json(raw_response)
        story.topic = cat
        
        print(f"✅ Generated: '{story.title}' with {len(story.scenes)} scenes")
        
        return story
    
    def save_story(self, story: Story, output_path: Optional[Path] = None) -> Path:
        """Save story to JSON file"""
        if output_path is None:
            # Create timestamped filename
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"story_{timestamp}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(story.model_dump_json(indent=2))
        
        print(f"💾 Saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate story scripts for YouTube Shorts")
    parser.add_argument("--category", "-c", type=str, help="Topic category (mystery, horror, sci-fi, fantasy, inspirational, thriller)")
    parser.add_argument("--topic", "-t", type=str, help="Custom topic/premise override")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file path")
    parser.add_argument("--model", "-m", type=str, default=OLLAMA_MODEL, help=f"Ollama model (default: {OLLAMA_MODEL})")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = StoryGenerator(model=args.model)
    
    # Generate story
    try:
        story = generator.generate(
            category=args.category,
            topic_override=args.topic
        )
        
        # Save
        output_path = Path(args.output) if args.output else None
        saved_path = generator.save_story(story, output_path)
        
        # Print summary
        print("\n📋 Story Summary:")
        print(f"   Title: {story.title}")
        print(f"   Scenes: {len(story.scenes)}")
        print(f"   Duration: {story.total_duration}s")
        if story.character_reference:
            print(f"   Character: {story.character_reference[:50]}...")
        
        return str(saved_path)
        
    except ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
