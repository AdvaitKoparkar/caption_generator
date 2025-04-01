"""
Database models and functions for caption generation and storage.
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, UTC
import json
import requests
from config_loader import config

db = SQLAlchemy()

class CaptionGeneration(db.Model):
    """Model for storing caption generations."""
    id = db.Column(db.Integer, primary_key=True)
    image_hash = db.Column(db.String(32), nullable=False)
    user_suggestion = db.Column(db.Text)
    model_metadata = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC))
    captions = db.relationship('Caption', backref='generation', lazy=True)
    comparisons = db.relationship('CaptionComparison', backref='generation', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'image_hash': self.image_hash,
            'user_suggestion': self.user_suggestion,
            'model_metadata': json.loads(self.model_metadata),
            'created_at': self.created_at.isoformat(),
            'captions': [caption.to_dict() for caption in self.captions]
        }

class Caption(db.Model):
    """Model for storing individual captions."""
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    is_selected = db.Column(db.Boolean, default=False)
    selected_at = db.Column(db.DateTime)
    generation_id = db.Column(db.Integer, db.ForeignKey('caption_generation.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC))

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'is_selected': self.is_selected,
            'selected_at': self.selected_at.isoformat() if self.selected_at else None,
            'generation_id': self.generation_id,
            'created_at': self.created_at.isoformat()
        }

class CaptionComparison(db.Model):
    """Model for storing caption preference comparisons."""
    id = db.Column(db.Integer, primary_key=True)
    generation_id = db.Column(db.Integer, db.ForeignKey('caption_generation.id'), nullable=False)
    positive_caption_id = db.Column(db.Integer, db.ForeignKey('caption.id'), nullable=False)
    negative_caption_id = db.Column(db.Integer, db.ForeignKey('caption.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC))

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

def save_caption_generation(image_hash: str, user_suggestion: str, descriptions: list, model_metadata: dict) -> CaptionGeneration:
    """
    Save a caption generation to the database.
    
    Args:
        image_hash: Hash of the image
        user_suggestion: User's suggestion for caption generation
        descriptions: List of generated captions
        model_metadata: Metadata about the model used
        
    Returns:
        CaptionGeneration object
    """
    generation = CaptionGeneration(
        image_hash=image_hash,
        user_suggestion=user_suggestion,
        model_metadata=json.dumps(model_metadata)
    )
    db.session.add(generation)
    db.session.flush()  # Get the generation ID
    
    for text in descriptions:
        caption = Caption(text=text, generation_id=generation.id)
        db.session.add(caption)
    
    db.session.commit()
    return generation

def update_caption_selection(caption_id: int) -> Caption:
    """
    Update the selected caption and create pairwise comparisons in the database.
    
    Args:
        caption_id: ID of the selected caption
        
    Returns:
        The selected caption object
    """
    selected_caption = Caption.query.get_or_404(caption_id)
    selected_caption.is_selected = True
    selected_caption.selected_at = datetime.now(UTC)
    
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

def get_selected_captions() -> list:
    """
    Get all selected captions from the database.
    
    Returns:
        List of selected captions with their metadata
    """
    selected_captions = Caption.query.filter_by(is_selected=True).all()
    return [caption.to_dict() for caption in selected_captions]

