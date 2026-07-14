import os
from dotenv import load_dotenv
import cohere

load_dotenv()
co = cohere.Client(os.getenv("EMBEDDING_API_KEY"))

response = co.embed(
    texts=["ধানের বাদামী দাগ রোগের কারণ কী?"],  # Bangla test string
    input_type="search_query",
    model="embed-multilingual-v3.0",
)
print(len(response.embeddings[0]))  # should print 1024 (the vector dimension)