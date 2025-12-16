# Pydantic Models for Data Validation
from typing import List, Optional
from pydantic import BaseModel, Field


class Scene(BaseModel):
    """Single scene in a story"""
    scene_id: int
    duration_seconds: int = Field(default=10, ge=5, le=20)
    visual_prompt: str = Field(..., description="Detailed prompt for Z-Image generation")
    narration: str = Field(..., description="Text to be spoken by TTS")
    character_description: Optional[str] = Field(None, description="Character details for consistency")


class Part(BaseModel):
    """One-minute part of a long-form story (contains multiple scenes)"""
    part_id: int
    part_title: str = Field(..., description="Title for this part/chapter")
    scenes: List[Scene]
    cliffhanger: Optional[str] = Field(None, description="Hook to keep viewers watching")
    
    @property
    def duration_seconds(self) -> int:
        return sum(s.duration_seconds for s in self.scenes)


class Story(BaseModel):
    """Complete story - supports both short-form (scenes) and long-form (parts)"""
    title: str
    topic: str
    
    # Long-form: 20 parts, each ~1 minute with multiple scenes
    parts: Optional[List[Part]] = Field(None, description="For long-form: 20 x 1-min parts")
    
    # Short-form: Direct scenes (legacy support for 60-second shorts)
    scenes: Optional[List[Scene]] = Field(None, description="For short-form: 6 scenes")
    
    total_duration: int = Field(default=1200, description="Total duration in seconds")
    synopsis: Optional[str] = Field(None, description="Story synopsis for long-form")
    character_reference: Optional[str] = Field(None, description="Main character description for consistency")
    
    @property
    def is_longform(self) -> bool:
        return self.parts is not None and len(self.parts) > 0
    
    @property
    def all_scenes(self) -> List[Scene]:
        """Get all scenes regardless of story type"""
        if self.is_longform:
            return [scene for part in self.parts for scene in part.scenes]
        return self.scenes or []
    
    def validate_duration(self) -> bool:
        """Check if scenes add up to target duration"""
        actual = sum(s.duration_seconds for s in self.all_scenes)
        tolerance = self.total_duration * 0.1  # 10% tolerance
        return abs(actual - self.total_duration) <= tolerance


class VisualAsset(BaseModel):
    """Generated visual assets for a scene"""
    scene_id: int
    part_id: Optional[int] = None  # For long-form videos
    base_image_path: str
    consistent_image_path: Optional[str] = None
    video_clip_path: Optional[str] = None


class AudioAsset(BaseModel):
    """Generated audio for a scene"""
    scene_id: int
    part_id: Optional[int] = None  # For long-form videos
    audio_path: str
    duration_seconds: float


class FinalVideo(BaseModel):
    """Final assembled video metadata"""
    output_path: str
    duration_seconds: float
    resolution: str
    title: str
    chapters: Optional[List[dict]] = None  # For YouTube chapters

