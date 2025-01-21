import os
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTModel

# Initialize the feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

def image_to_text(image_path):
    # Load the image using PIL
    img = Image.open(image_path)

    # Preprocess the image
    inputs = feature_extractor(images=[img], return_tensors="pt")

    # Forward pass with model
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state

    # Get the last token in the sequence (text start symbol)
    text_start_token_index = torch.argmax(last_hidden_state[:, 0, :])  # BERT-like approach for text classification

    # Extract the generated text from the output
    text = feature_extractor.decode(ids=torch.argmax(last_hidden_state[:, 0, :], dim=-1), skip_special_tokens=True)

    return text

# Test the function with an image path
image_path = "test-image.png"
print(image_to_text(image_path))