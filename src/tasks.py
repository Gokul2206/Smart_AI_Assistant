# tasks.py — Basic AI tasks: Model Recommender + Code Explainer

from google import genai
from src.prompts import MODEL_RECOMMENDER_PROMPT, CODE_EXPLAINER_PROMPT
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


def get_model_recommendation(problem: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    prompt = MODEL_RECOMMENDER_PROMPT.format(problem=problem)
    response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
    return response.text

def explain_ml_code(code: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    prompt = CODE_EXPLAINER_PROMPT.format(code=code)
    response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
    return response.text