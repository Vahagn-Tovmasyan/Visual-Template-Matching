import os
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Ensure API key is set
from dotenv import load_dotenv
load_dotenv()

from src.hybrid_dino import detect_hybrid_dino
from src.hybrid import _verify_crop_with_vlm
from src.schema import DetectionResult

def main():
    template = "test_images/000000017029_Template.jpg"
    scene = "test_images/Generated Image March 30, 2026 - 4_20PM.jpg"
    
    # Check if files exist
    if not os.path.exists(template) or not os.path.exists(scene):
        print(f"Skipping because files do not exist.")
        scene = "test_images/Generated Image.jpg"
        if not os.path.exists(scene):
            return

    print("Running detect_hybrid_dino...")
    
    # We will pass a very low threshold to mimic Gradio drop
    result = detect_hybrid_dino(
        template_path=template,
        scene_path=scene,
        confidence_threshold=0.35,  # VLM threshold
        dino_threshold=0.2,         # DINO threshold
    )
    
    print("\n\nFINAL RESULT:")
    print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
