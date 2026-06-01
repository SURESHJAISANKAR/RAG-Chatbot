from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

def generate_answer(question: str, context: str):
    prompt = f"""
    Answer the question based ONLY on the context below.
    If answer is not found, say "I don't know".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    return response.choices[0].message.content