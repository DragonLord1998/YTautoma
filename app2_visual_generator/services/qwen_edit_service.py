"""
Qwen-Image-Edit Service (Local)
Character consistency using Qwen-Image-Edit model.
https://github.com/QwenLM/Qwen-Image
"""

import torch
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import QWEN_IMAGE_EDIT_MODEL, TORCH_DTYPE, LOW_VRAM_MODE


class QwenImageEditService:
    """Apply character consistency using local Qwen-Image-Edit model"""
    
    def __init__(self, model_id: str = QWEN_IMAGE_EDIT_MODEL):
        """
        Initialize Qwen-Image-Edit service.
        
        Args:
            model_id: HuggingFace model ID or local path
        """
        self.model_id = model_id
        self.pipe = None
        self._loaded = False
    
    def load_model(self):
        """Load the Qwen-Image-Edit pipeline"""
        if self._loaded:
            return
        
        print(f"🔄 Loading Qwen-Image-Edit: {self.model_id}")
        print("   This requires significant VRAM (40GB+ recommended)...")
        
        try:
            # Qwen-Image-Edit uses a custom pipeline
            # Note: This may require specific installation from Qwen-Image repo
            from transformers import AutoProcessor, AutoModel
            from PIL import Image
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=getattr(torch, TORCH_DTYPE),
                trust_remote_code=True,
                device_map="auto" if LOW_VRAM_MODE else None
            )
            
            if not LOW_VRAM_MODE:
                self.model = self.model.cuda()
            
            self._loaded = True
            print("✅ Qwen-Image-Edit loaded")
            
        except Exception as e:
            print(f"⚠️ Failed to load Qwen-Image-Edit: {e}")
            print("   Character consistency will be skipped.")
            self._loaded = False
    
    def apply_consistency(
        self,
        source_image_path: Path,
        reference_image_path: Path,
        prompt: str,
        character_description: str
    ):
        """
        Apply character from reference to source image.
        
        Args:
            source_image_path: New scene image
            reference_image_path: Reference character image
            prompt: Scene description
            character_description: Character details for this scene
            
        Returns:
            PIL.Image with consistent character
        """
        if not self._loaded:
            self.load_model()
        
        if not self._loaded:
            # Model failed to load, return source as-is
            from PIL import Image
            return Image.open(source_image_path)
        
        from PIL import Image
        
        source_img = Image.open(source_image_path)
        reference_img = Image.open(reference_image_path)
        
        # Build edit prompt for consistency
        edit_prompt = (
            f"Edit this image to match the character from the reference image. "
            f"Character: {character_description}. "
            f"Scene: {prompt}. "
            f"Keep the same face, clothing, and appearance as the reference."
        )
        
        print(f"🔄 Applying character consistency...")
        
        try:
            # Process inputs
            inputs = self.processor(
                text=edit_prompt,
                images=[source_img, reference_img],
                return_tensors="pt"
            )
            
            if torch.cuda.is_available() and not LOW_VRAM_MODE:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            
            # Decode output image
            result_image = self.processor.decode(outputs)
            
            print("✅ Character consistency applied")
            return result_image
            
        except Exception as e:
            print(f"⚠️ Consistency edit failed: {e}")
            print("   Returning original image")
            return source_img
    
    def apply_to_file(
        self,
        source_image: Path,
        reference_image: Path,
        output_path: Path,
        prompt: str,
        character_description: str
    ) -> Path:
        """Apply consistency and save to file"""
        result = self.apply_consistency(
            source_image,
            reference_image,
            prompt,
            character_description
        )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        
        print(f"💾 Saved to: {output_path}")
        return output_path
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self._loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("🧹 Qwen-Image-Edit unloaded")


class SimpleConsistencyService:
    """
    Fallback consistency service using image blending.
    Used when Qwen-Image-Edit is not available.
    """
    
    def __init__(self):
        self._loaded = True
    
    def load_model(self):
        pass
    
    def apply_consistency(
        self,
        source_image_path: Path,
        reference_image_path: Path,
        prompt: str,
        character_description: str
    ):
        """Simple fallback - returns source image as-is"""
        from PIL import Image
        print("⚠️ Using fallback consistency (no editing)")
        return Image.open(source_image_path)
    
    def apply_to_file(
        self,
        source_image: Path,
        reference_image: Path,
        output_path: Path,
        **kwargs
    ) -> Path:
        """Copy source to output (no actual editing)"""
        from PIL import Image
        img = Image.open(source_image)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return output_path
