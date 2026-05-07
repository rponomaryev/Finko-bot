import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
KB_FOLDER = BASE_DIR / "kb"

load_dotenv(dotenv_path=ENV_FILE)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it first.")

client = OpenAI(api_key=api_key)

FILES_TO_UPLOAD = [
    "finko_knowledge_ru.txt",
    "finko_knowledge_en.txt",
    "finko_knowledge_uz.txt",
    "finko_knowledge_uz_cyrl.txt",
]


def main() -> None:
    print("Создаем Vector Store...")

    vector_store = client.vector_stores.create(name="Finko Knowledge Base")
    print(f"Vector Store создан: {vector_store.id}")

    file_ids: list[str] = []

    for filename in FILES_TO_UPLOAD:
        file_path = KB_FOLDER / filename

        if not file_path.exists():
            print(f"Файл не найден: {file_path}")
            continue

        print(f"Загружаю файл: {filename}")
        with file_path.open("rb") as f:
            uploaded_file = client.files.create(file=f, purpose="assistants")
        print(f"Файл загружен: {uploaded_file.id}")
        file_ids.append(uploaded_file.id)

    if not file_ids:
        raise RuntimeError("Нет файлов для загрузки. Проверь папку kb/.")

    print("Добавляю файлы в Vector Store...")
    batch = client.vector_stores.file_batches.create(
        vector_store_id=vector_store.id,
        file_ids=file_ids,
    )
    print(f"Batch создан: {batch.id}")

    while True:
        status = client.vector_stores.file_batches.retrieve(
            vector_store_id=vector_store.id,
            batch_id=batch.id,
        )

        print("Статус:", status.status)

        if status.status in {"completed", "failed", "cancelled"}:
            print("Итоговый статус:", status.status)
            print("Детали:", status.file_counts)
            break

        time.sleep(3)

    if status.status != "completed":
        raise RuntimeError(f"Vector Store batch finished with status: {status.status}")

    print("\nГотово.")
    print(f"OPENAI_VECTOR_STORE_ID={vector_store.id}")


if __name__ == "__main__":
    main()