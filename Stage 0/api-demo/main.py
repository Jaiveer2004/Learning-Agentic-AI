import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key = os.getenv("API_KEY"))

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Hi, I am Jaiveer. What is Your Name? Say Hello to me."
)

print(interaction.output_text)