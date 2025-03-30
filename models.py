import torch
from PIL import Image
from GenerativeImage2Text.model import GiT
import requests
import json
from functools import lru_cache
import hashlib
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask import current_app

db = SQLAlchemy()

class CaptionGeneration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_hash = db.Column(db.String(32), nullable=False)
    user_suggestion = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    model_metadata = db.Column(db.JSON)
    captions = db.relationship('Caption', backref='generation', lazy=True)
    comparisons = db.relationship('CaptionComparison', backref='generation', lazy=True)

class Caption(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    generation_id = db.Column(db.Integer, db.ForeignKey('caption_generation.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    is_selected = db.Column(db.Boolean, default=False)
    selected_at = db.Column(db.DateTime)

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'is_selected': self.is_selected,
            'selected_at': self.selected_at.isoformat() if self.selected_at else None
        }

class CaptionComparison(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    generation_id = db.Column(db.Integer, db.ForeignKey('caption_generation.id'), nullable=False)
    positive_caption_id = db.Column(db.Integer, db.ForeignKey('caption.id'), nullable=False)
    negative_caption_id = db.Column(db.Integer, db.ForeignKey('caption.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    positive_caption = db.relationship('Caption', foreign_keys=[positive_caption_id], backref='positive_comparisons')
    negative_caption = db.relationship('Caption', foreign_keys=[negative_caption_id], backref='negative_comparisons')

    def to_dict(self):
        return {
            'id': self.id,
            'generation_id': self.generation_id,
            'positive_caption': self.positive_caption.to_dict(),
            'negative_caption': self.negative_caption.to_dict(),
            'created_at': self.created_at.isoformat()
        }

# Global variables for lazy loading
device = "cuda" if torch.cuda.is_available() else "cpu"
git_model = None

def get_git_model():
    """
    Lazy load the GiT model only when needed.
    """
    global git_model
    if git_model is None:
        git_model = GiT(
            checkpoint_dir="checkpoints",
            run_name="latest",
            device=device
        )
    return git_model

@lru_cache(maxsize=100)
def describe_image(image_hash):
    """
    Generate captions for an image using the GiT model with caching.
    
    Args:
        image_hash: Hash of the image for caching
        
    Returns:
        List of generated captions
    """
    git_model = get_git_model()
    return git_model.describe_images(image_hash)

def get_image_hash(image):
    """
    Generate a hash for the image to use as cache key.
    """
    # Convert image to bytes
    img_byte_arr = image.tobytes()
    return hashlib.md5(img_byte_arr).hexdigest()

def enhance_description(image, user_suggestion=None):
    """
    Generate descriptions using GiT and enhance each one using Llama 2 via Ollama.
    
    Args:
        image: PIL Image object
        user_suggestion: Optional string containing user's suggestion for caption style
        
    Returns:
        List of enhanced descriptions
    """
    # Get image hash for caching
    image_hash = get_image_hash(image)
    
    # Get initial descriptions from GiT (cached)
    descriptions = describe_image(image_hash)
    enhanced_descriptions = []
    
    # Enhance each description using Llama 2
    for initial_description in descriptions:
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
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2",
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
            enhanced_descriptions.append(caption)
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            enhanced_descriptions.append(initial_description)  # Fallback to original description if API call fails
    
    return enhanced_descriptions

def save_caption_generation(image_hash, user_suggestion, captions, model_metadata):
    """
    Save caption generation data to the database.
    
    Args:
        image_hash: Hash of the uploaded image
        user_suggestion: User's style suggestion
        captions: List of generated captions
        model_metadata: Dictionary containing model information
        
    Returns:
        CaptionGeneration object
    """
    generation = CaptionGeneration(
        image_hash=image_hash,
        user_suggestion=user_suggestion,
        model_metadata=model_metadata
    )
    db.session.add(generation)
    db.session.flush()  # Get the generation ID
    
    # Add captions
    for caption_text in captions:
        caption = Caption(
            generation_id=generation.id,
            text=caption_text
        )
        db.session.add(caption)
    
    db.session.commit()
    return generation

def update_caption_selection(caption_id):
    """
    Update the selected caption and create pairwise comparisons in the database.
    
    Args:
        caption_id: ID of the selected caption
    """
    selected_caption = Caption.query.get_or_404(caption_id)
    selected_caption.is_selected = True
    selected_caption.selected_at = datetime.utcnow()
    
    # Get all other captions from the same generation
    other_captions = Caption.query.filter(
        Caption.generation_id == selected_caption.generation_id,
        Caption.id != caption_id
    ).all()
    
    # Create pairwise comparisons
    for other_caption in other_captions:
        comparison = CaptionComparison(
            generation_id=selected_caption.generation_id,
            positive_caption_id=selected_caption.id,
            negative_caption_id=other_caption.id
        )
        db.session.add(comparison)
    
    db.session.commit()
    return selected_caption

