import torch
from PIL import Image
from transformers import T5Tokenizer, T5ForConditionalGeneration
from GenerativeImage2Text.model import GiT
import requests
import json

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize GiT model
git_model = GiT(
    checkpoint_dir="checkpoints",
    run_name="latest",
    device=device
)

# Load a small language model for modifications
caption_modifier_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=True)
caption_modifier_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

def describe_image(image):
    """
    Generate captions for an image using the GiT model.
    
    Args:
        image: PIL Image object
        
    Returns:
        List of generated captions
    """
    return git_model.describe_images(image)

def enhance_description(image):
    """
    Generate descriptions using GiT and enhance each one using Llama 2 via Ollama.
    
    Args:
        image: PIL Image object
        
    Returns:
        List of enhanced descriptions
    """
    # Get initial descriptions from GiT
    descriptions = git_model.describe_images(image)
    enhanced_descriptions = []
    
    # Enhance each description using Llama 2
    for initial_description in descriptions:
        # Prepare the prompt for Llama 2
        prompt = f"""Based on this image description: "{initial_description}"
        Please provide a more detailed and engaging description that captures the essence of the image.
        Focus on visual elements, atmosphere, and any notable details.
        Make it more vivid and descriptive while maintaining accuracy."""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2",
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()
            enhanced_descriptions.append(result["response"])
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            enhanced_descriptions.append(initial_description)  # Fallback to original description if API call fails
    
    return enhanced_descriptions

