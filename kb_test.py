import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")

def main():
    if not VECTOR_STORE_ID:
        print("OPENAI_VECTOR_STORE_ID не найден в .env")
        return

    query = "Где находится Finko?"

    result = client.vector_stores.search(
        vector_store_id=VECTOR_STORE_ID,
        query=query,
        max_num_results=3
    )

    print(f"\nЗапрос: {query}")
    print("\nРЕЗУЛЬТАТЫ ПОИСКА:\n")

    data = getattr(result, "data", []) or []
    if not data:
        print("Ничего не найдено")
        return

    for i, item in enumerate(data, start=1):
        filename = getattr(item, "filename", None) or getattr(item, "file_name", None) or "unknown_file"
        print(f"\n=== RESULT {i} | FILE: {filename} ===")

        content_list = getattr(item, "content", []) or []
        found_text = False

        for block in content_list:
            text = getattr(block, "text", None)
            if text:
                found_text = True
                print(text.strip())
                print("-" * 80)

        if not found_text:
            print("В этом результате нет текста")

if __name__ == "__main__":
    main()