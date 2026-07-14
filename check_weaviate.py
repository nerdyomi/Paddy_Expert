import weaviate, os
from dotenv import load_dotenv

load_dotenv()
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
)
print(client.is_ready())  # should print True
client.close()