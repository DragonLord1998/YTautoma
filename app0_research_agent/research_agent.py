#!/usr/bin/env python3
"""
App 0: Research Agent
Deep research and trending topic discovery for long-form video content.
Uses local LLM (36B+ parameters) with optional web search tools.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from shared.config import (
    OLLAMA_BASE_URL,
    RESEARCH_MODEL,
    SEARXNG_URL,
    TOTAL_PARTS
)


@dataclass
class ResearchResult:
    """Structured research output for story generation"""
    original_idea: str
    refined_topic: str
    key_themes: List[str]
    plot_suggestions: List[str]
    character_archetypes: List[str]
    trending_angles: List[str]
    target_audience: str
    emotional_journey: str
    research_sources: List[Dict[str, str]]


class WebSearchTool:
    """Web search using SearXNG or fallback methods"""
    
    def __init__(self, searxng_url: str = SEARXNG_URL):
        self.searxng_url = searxng_url
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if SearXNG is available"""
        try:
            # Try a simple search to verify SearXNG is working
            response = requests.get(
                f"{self.searxng_url}/search",
                params={"q": "test", "format": "json"},
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            print("⚠️ SearXNG not available, using LLM knowledge only")
            return False
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Search the web for relevant content"""
        if not self.available:
            return []
        
        try:
            response = requests.get(
                f"{self.searxng_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "general",
                    "engines": "google,bing,duckduckgo"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get("results", [])[:num_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", "")[:300]
                    })
                return results
        except Exception as e:
            print(f"⚠️ Search error: {e}")
        
        return []


class ResearchAgent:
    """
    Deep research agent for content ideation.
    Uses local LLM with web search tools to discover trending topics
    and structure compelling 20-part narratives.
    """
    
    def __init__(
        self,
        model: str = RESEARCH_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        enable_web_search: bool = True
    ):
        self.model = model
        self.base_url = base_url
        self.web_search = WebSearchTool() if enable_web_search else None
    
    def _call_ollama(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """Call Ollama API with the given prompts"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": 4000
            }
        }
        
        print(f"🔬 Researching with {self.model}...")
        
        try:
            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: 'ollama serve'"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("Research request timed out.")
    
    def _gather_web_context(self, idea: str) -> str:
        """Gather relevant web context for the idea"""
        if not self.web_search or not self.web_search.available:
            return ""
        
        print("🌐 Searching for trending topics...")
        
        # Search for different aspects
        searches = [
            f"{idea} trending 2024",
            f"{idea} viral stories",
            f"{idea} documentary style",
        ]
        
        all_results = []
        for query in searches:
            results = self.web_search.search(query, num_results=3)
            all_results.extend(results)
        
        if not all_results:
            return ""
        
        context = "## Web Research Results\n\n"
        for i, result in enumerate(all_results[:9], 1):
            context += f"{i}. **{result['title']}**\n"
            context += f"   {result['snippet']}\n\n"
        
        return context
    
    def research(self, user_idea: str) -> ResearchResult:
        """
        Conduct deep research on user's idea.
        
        Args:
            user_idea: The user's initial content idea
            
        Returns:
            ResearchResult with structured research output
        """
        print(f"\n🔍 Researching: {user_idea}")
        print("=" * 50)
        
        # Gather web context if available
        web_context = self._gather_web_context(user_idea)
        
        # Build research prompt
        system_prompt = """You are an expert content researcher and storyteller. 
Your task is to take a user's idea and transform it into a compelling 20-minute video concept.

You must return ONLY valid JSON matching this structure:
{
    "refined_topic": "A refined, compelling version of the idea",
    "key_themes": ["theme1", "theme2", "theme3"],
    "plot_suggestions": ["20-part narrative arc suggestions"],
    "character_archetypes": ["main character types that would work"],
    "trending_angles": ["current trends that connect to this topic"],
    "target_audience": "Description of ideal viewer",
    "emotional_journey": "The emotional arc across 20 parts"
}

Make it VIRAL-WORTHY. Think about what makes people binge-watch content."""
        
        user_prompt = f"""Research this idea for a 20-minute video (20 x 1-minute parts):

**User's Idea:** {user_idea}

{web_context}

Create a research report that will help generate an AMAZING {TOTAL_PARTS}-part story.
Focus on:
1. What makes this topic compelling NOW
2. Unique angles that haven't been explored
3. Emotional hooks that keep viewers watching
4. Character types that resonate with audiences

Return ONLY the JSON, no other text."""
        
        # Call LLM for research
        raw_response = self._call_ollama(system_prompt, user_prompt)
        
        # Parse response
        research_data = self._parse_research_json(raw_response)
        
        return ResearchResult(
            original_idea=user_idea,
            refined_topic=research_data.get("refined_topic", user_idea),
            key_themes=research_data.get("key_themes", []),
            plot_suggestions=research_data.get("plot_suggestions", []),
            character_archetypes=research_data.get("character_archetypes", []),
            trending_angles=research_data.get("trending_angles", []),
            target_audience=research_data.get("target_audience", "General audience"),
            emotional_journey=research_data.get("emotional_journey", ""),
            research_sources=[{"type": "llm", "model": self.model}]
        )
    
    def _parse_research_json(self, raw_response: str) -> Dict[str, Any]:
        """Parse LLM response into dictionary"""
        cleaned = raw_response.strip()
        
        # Remove markdown code blocks if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            cleaned = "\n".join(lines)
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error: {e}")
            # Return minimal valid structure
            return {
                "refined_topic": "Research parsing failed",
                "key_themes": [],
                "plot_suggestions": [],
                "character_archetypes": [],
                "trending_angles": [],
                "target_audience": "General",
                "emotional_journey": ""
            }
    
    def save_research(self, result: ResearchResult, output_path: Path) -> Path:
        """Save research result to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "original_idea": result.original_idea,
            "refined_topic": result.refined_topic,
            "key_themes": result.key_themes,
            "plot_suggestions": result.plot_suggestions,
            "character_archetypes": result.character_archetypes,
            "trending_angles": result.trending_angles,
            "target_audience": result.target_audience,
            "emotional_journey": result.emotional_journey,
            "research_sources": result.research_sources
        }
        
        output_path.write_text(json.dumps(data, indent=2))
        print(f"💾 Research saved: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Research agent for content ideation")
    parser.add_argument("--idea", "-i", type=str, required=True, help="Your content idea")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file path")
    parser.add_argument("--model", "-m", type=str, default=RESEARCH_MODEL, help=f"Ollama model (default: {RESEARCH_MODEL})")
    parser.add_argument("--no-web", action="store_true", help="Disable web search")
    
    args = parser.parse_args()
    
    agent = ResearchAgent(
        model=args.model,
        enable_web_search=not args.no_web
    )
    
    try:
        result = agent.research(args.idea)
        
        print("\n📋 Research Summary:")
        print(f"   Topic: {result.refined_topic}")
        print(f"   Themes: {', '.join(result.key_themes[:3])}")
        print(f"   Audience: {result.target_audience}")
        
        if args.output:
            agent.save_research(result, Path(args.output))
        
        return result
        
    except Exception as e:
        print(f"❌ Research failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
