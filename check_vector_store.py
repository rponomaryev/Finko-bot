from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

vs_id = os.getenv("OPENAI_VECTOR_STORE_ID")

print("Using VS ID:", vs_id)

store = client.vector_stores.retrieve(vs_id)
print("Vector store:", store.id)
print("Status:", store.status)
print("File counts:", store.file_counts)

files = client.vector_stores.files.list(vs_id)

print("\nFiles:")
for f in files.data:
    print(f.id, f.status, f.last_error)