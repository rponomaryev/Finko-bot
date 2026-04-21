import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

KB_FOLDER = Path("kb")

FILES_TO_UPLOAD = [
    "faq_b2b.docx",
    "faq_clients.doc",
    "site_about.txt",
    "site_faq.txt",
    "site_products.txt",
]

def main():
    print("Создаем Vector Store...")

    vector_store = client.vector_stores.create(
        name="Finko Knowledge Base"
    )
    print(f"Vector Store создан: {vector_store.id}")

    file_ids = []

    for filename in FILES_TO_UPLOAD:
        file_path = KB_FOLDER / filename

        if not file_path.exists():
            print(f"Файл не найден: {file_path}")
            continue

        print(f"Загружаю файл: {filename}")
        with open(file_path, "rb") as f:
            uploaded_file = client.files.create(
                file=f,
                purpose="assistants"
            )
        print(f"Файл загружен: {uploaded_file.id}")
        file_ids.append(uploaded_file.id)

    if not file_ids:
        print("Нет файлов для загрузки")
        return

    print("Добавляю файлы в Vector Store...")
    batch = client.vector_stores.file_batches.create(
        vector_store_id=vector_store.id,
        file_ids=file_ids
    )
    print(f"Batch создан: {batch.id}")

    while True:
        status = client.vector_stores.file_batches.retrieve(
            vector_store_id=vector_store.id,
            batch_id=batch.id
        )

        print("Статус:", status.status)

        if status.status in ["completed", "failed", "cancelled"]:
            print("Итоговый статус:", status.status)
            print("Детали:", status.file_counts)
            break

        time.sleep(3)

    print("\nГотово.")
    print(f"OPENAI_VECTOR_STORE_ID={vector_store.id}")

if __name__ == "__main__":
    main()