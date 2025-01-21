import os
import tensor
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline

# load_dotenv(find_dotenv())

# API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# img2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]["generated_text"]

    print(text)
    return text

img2text("test-image.png")

# llm
# https://huggingface.co/spaces/KwaiVGI/LivePortrait

# text to speech
