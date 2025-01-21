import os
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import InferenceClient

load_dotenv(find_dotenv())

API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(api_key=API_KEY)

messages = [
	{
		"role": "user",
		"content": [
			{
				"type": "text",
				"text": "Describe this image in one sentence."
			},
			{
				"type": "image_url",
				"image_url": {
					"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
				}
			}
		]
	}
]

# img2text
completion = client.chat.completions.create(
    model="meta-llama/Llama-3.2-11B-Vision-Instruct", 
	messages=messages, 
	max_tokens=500
)

print(completion.choices[0].message)


# llm


# text to speech
