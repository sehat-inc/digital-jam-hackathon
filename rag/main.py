import google.generativeai as genai
from dotenv import load_dotenv
import os 

load_dotenv()

# Getting API Key
genai.configure(api_key=os.getenv('GEMINI_API'))

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Who are you?")
print(response.text)
