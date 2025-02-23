import torch
from PIL import Image
from transformers import pipeline
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# model to generate captions from image with prompt
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load a small language model for modifications
caption_modifier_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=True)
caption_modifier_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

# Function to generate captions
def generate_captions(image, prompt=""):
    prompt = f"{prompt}. Do not include any proper nouns other than the ones explicitly included in the give caption."
    inputs = blip_processor(images=image, test=prompt, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs, max_length=30, num_return_sequences=5, do_sample=True)
    return [blip_processor.decode(caption, skip_special_tokens=True) for caption in output]

# Function to rank captions using CLIP
def rank_captions(image, captions):
    clip_inputs = clip_processor(text=captions, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        clip_outputs = clip_model(**clip_inputs)
        logits_per_image = clip_outputs.logits_per_image    
        similarity_scores = logits_per_image.softmax(dim=1).detach().numpy().flatten()
    
    sorted_indices = similarity_scores.argsort()[::-1]
    ranked_captions = [captions[i] for i in sorted_indices]
    
    return ranked_captions[:5]  # Return top 5 captions

def modify_caption(selected_caption, user_instruction):
    prompt = f"Rewrite this caption: '{selected_caption}' based on user instruction: {user_instruction}. Do not include any names other than the ones already in the selected caption."
    print(prompt)
    input_ids = caption_modifier_tokenizer(prompt, return_tensors="pt")
    outputs = caption_modifier_model.generate(max_new_tokens=input_ids['input_ids'].shape[-1], **input_ids)
    modified_caption = caption_modifier_tokenizer.decode(outputs[0])
    return modified_caption