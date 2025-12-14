# Pydantic Models for Data Validation
from typing import List, Optional
from pydantic import BaseModel, Field


class Scene(BaseModel):
    """Single scene in a story"""
    scene_id: int
    duration_seconds: int = Field(default=10, ge=5, le=15)
    visual_prompt: str = Field(..., description="Detailed prompt for Z-Image generation")
    narration: str = Field(..., description="Text to be spoken by TTS")
    character_description: Optional[str] = Field(None, description="Character details for consistency")


class Story(BaseModel):
    """Complete story with all scenes"""
    title: str
    topic: str
    scenes: List[Scene]
    total_duration: int = Field(default=60)
    character_reference: Optional[str] = Field(None, description="Main character description for consistency")
    
    def validate_duration(self) -> bool:
        """Check if scenes add up to target duration"""
        actual = sum(s.duration_seconds for s in self.scenes)
        return abs(actual - self.total_duration) <= 5  # 5 second tolerance


class VisualAsset(BaseModel):
    """Generated visual assets for a scene"""
    scene_id: int
    base_image_path: str
    consistent_image_path: Optional[str] = None
    video_clip_path: Optional[str] = None


class AudioAsset(BaseModel):
    """Generated audio for a scene"""
    scene_id: int
    audio_path: str
    duration_seconds: float


class FinalVideo(BaseModel):
    """Final assembled video metadata"""
    output_path: str
    duration_seconds: float
    resolution: str
    title: str
