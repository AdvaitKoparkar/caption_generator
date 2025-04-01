"""
FastAPI service for the GiT model.
Provides endpoints for generating image captions.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from typing import List
from pydantic import BaseModel
import torch
import time
from collections import deque
from datetime import datetime
import requests
import logging
import uvicorn

from .GiT import GiT
from config_loader import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GiT Caption Generator API",
    description="API for generating image captions using the GiT model",
    version="1.0.0"
)

# Performance metrics
latency_history = deque(maxlen=100)  # Store last 100 latencies
request_count = 0
start_time = datetime.now()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize model
device = config.git_config.device
if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

try:
    model = GiT(
        checkpoint_dir=config.git_config.checkpoint_dir,
        run_name=config.git_config.run_name,
        device=device
    )
    logger.info("GiT model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize GiT model: {e}")
    raise

class CaptionRequest(BaseModel):
    suggestion: str = ""

class CaptionResponse(BaseModel):
    captions: List[str]

def enhance_with_llama(initial_description: str, user_suggestion: str = "") -> str:
    """
    Enhance a caption using Llama 2 via Ollama.
    
    Args:
        initial_description: Initial caption from GiT
        user_suggestion: User's style suggestion
        
    Returns:
        Enhanced caption
    """
    # Prepare the prompt for Llama 2
    base_prompt = f"""Based on this image description: "{initial_description}"
    Generate a single social media caption. The caption should be:
    - No longer than 200 characters
    - Engaging and appropriate for platforms like Instagram, Twitter, and Facebook
    - Direct and concise (no explanations or additional text)
    - Start with the caption immediately (no prefixes like "Here's a caption:" or "Caption:")
    - No hashtags
    - No emojis
    - No special characters
    Caption:"""

    # Add user suggestion if provided
    if user_suggestion:
        prompt = f"{base_prompt}\n\nAdditional style instruction: {user_suggestion}"
    else:
        prompt = base_prompt

    try:
        ollama_url = f"http://{config.services['ollama'].host}:{config.services['ollama'].port}/api/generate"
        logger.info(f"Calling Ollama API at {ollama_url}")
        response = requests.post(
            ollama_url,
            json={
                "model": config.services['ollama'].model,
                "prompt": prompt,
                "stream": False
            },
            timeout=30  # Add timeout to prevent hanging
        )
        response.raise_for_status()
        result = response.json()
        # Clean up the response to ensure it's just the caption
        caption = result["response"].strip()
        # Remove any prefixes that might have been added
        caption = caption.replace("Caption:", "").replace("Here's a caption:", "").strip()
        return caption
    except Exception as e:
        logger.error(f"Error calling Ollama API: {e}")
        return initial_description  # Fallback to original description if API call fails

@app.post("/generate", response_model=CaptionResponse)
async def generate_caption(
    file: UploadFile = File(...),
    suggestion: str = ""
):
    """
    Generate captions for an uploaded image.
    
    Args:
        file: The image file to generate captions for
        suggestion: User's suggestion for caption style
        
    Returns:
        List of enhanced captions
    """
    global request_count
    request_count += 1
    
    try:
        logger.info("=== Starting caption generation ===")
        logger.info(f"Received file: {file.filename}")
        logger.info(f"User suggestion: {suggestion}")
        
        # Read and validate image
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes from file")
        image = Image.open(io.BytesIO(contents))
        logger.info(f"Successfully opened image: {file.filename}")
        
        # Generate initial captions with GiT
        logger.info("Starting GiT model caption generation...")
        start_time = time.time()
        initial_captions = model.describe_images(image)
        latency = time.time() - start_time
        latency_history.append(latency)
        logger.info(f"Generated {len(initial_captions)} initial captions in {latency:.2f}s")
        logger.info(f"Initial captions: {initial_captions}")
        
        # Enhance each caption with Llama
        logger.info("Starting Llama enhancement...")
        enhanced_captions = []
        for i, caption in enumerate(initial_captions):
            logger.info(f"Enhancing caption {i+1}/{len(initial_captions)}")
            enhanced_caption = enhance_with_llama(caption, suggestion)
            enhanced_captions.append(enhanced_caption)
            logger.info(f"Enhanced caption {i+1}: {enhanced_caption}")
        
        logger.info("Successfully generated all captions")
        logger.info(f"Final enhanced captions: {enhanced_captions}")
        logger.info("=== Caption generation complete ===")
        return CaptionResponse(captions=enhanced_captions)
        
    except Exception as e:
        logger.error(f"Error generating captions: {e}")
        logger.error("=== Caption generation failed ===")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    current_time = datetime.now()
    uptime = (current_time - start_time).total_seconds()
    
    return {
        "total_requests": request_count,
        "average_latency": sum(latency_history) / len(latency_history) if latency_history else 0,
        "uptime_seconds": uptime,
        "requests_per_second": request_count / uptime if uptime > 0 else 0
    }

def run_api():
    """
    Run the FastAPI service.
    """
    host = config.services['git_api'].host
    port = config.services['git_api'].port
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_api() 