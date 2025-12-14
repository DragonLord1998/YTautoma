"""
Z-Image Service (Local Diffusers)
Offline image generation using Z-Image-Turbo model.
https://github.com/Tongyi-MAI/Z-Image
"""

import torch
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import (
    ZIMAGE_MODEL, ZIMAGE_DEVICE, VIDEO_WIDTH, VIDEO_HEIGHT,
    TORCH_DTYPE, LOW_VRAM_MODE
)


class ZImageService:
    """Generate images using Z-Image-Turbo model locally via diffusers"""
    
    def __init__(
        self,
        model_id: str = ZIMAGE_MODEL,
        device: str = ZIMAGE_DEVICE,
        dtype: str = TORCH_DTYPE
    ):
        """
        Initialize Z-Image pipeline.
        
        Args:
            model_id: HuggingFace model ID or local path
            device: 'cuda' or 'cpu'
            dtype: 'bfloat16', 'float16', or 'float32'
        """
        self.model_id = model_id
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.pipe = None
        self._loaded = False
    
    def load_model(self):
        """Load the Z-Image pipeline (call explicitly due to memory)"""
        if self._loaded:
            return
        
        print(f"🔄 Loading Z-Image model: {self.model_id}")
        print("   This may take a while on first run...")
        
        try:
            from diffusers import ZImagePipeline
            
            self.pipe = ZImagePipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            self.pipe.to(self.device)
            
            # Memory optimizations
            if LOW_VRAM_MODE:
                print("   Enabling memory optimizations...")
                self.pipe.enable_model_cpu_offload()
            
            self._loaded = True
            print("✅ Z-Image loaded successfully")
            
        except ImportError:
            raise ImportError(
                "Z-Image requires latest diffusers. Install with:\n"
                "pip install git+https://github.com/huggingface/diffusers"
            )
    
    def generate(
        self,
        prompt: str,
        width: int = VIDEO_WIDTH,
        height: int = VIDEO_HEIGHT,
        seed: Optional[int] = None,
        num_inference_steps: int = 9,  # Turbo model uses 8-9 steps
        guidance_scale: float = 0.0,   # Turbo model uses 0 guidance
    ):
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image
            width: Image width
            height: Image height
            seed: Random seed for reproducibility
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale (0 for Turbo models)
            
        Returns:
            PIL.Image object
        """
        if not self._loaded:
            self.load_model()
        
        # Setup generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)
        
        print(f"🎨 Generating image...")
        print(f"   Prompt: {prompt[:60]}...")
        
        result = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        image = result.images[0]
        print("✅ Image generated")
        
        return image
    
    def generate_to_file(
        self,
        prompt: str,
        output_path: Path,
        **kwargs
    ) -> Path:
        """Generate image and save to file"""
        image = self.generate(prompt, **kwargs)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        
        print(f"💾 Saved to: {output_path}")
        return output_path
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self._loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("🧹 Z-Image unloaded")
