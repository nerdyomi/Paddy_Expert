import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # or load_dotenv("D:/paddy-rag/.env") if run from elsewhere

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    max_tokens=10,
)

print(response.choices[0].message.content)