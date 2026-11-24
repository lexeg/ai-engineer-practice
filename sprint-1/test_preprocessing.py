
import json
from collections import Counter

from openai import OpenAI
from transformers import AutoTokenizer

from run_batch import MODEL, SYSTEM_PROMPT


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)


PREFIXES = (
    "Пользуюсь пару недель. ",
    "Пользуюсь месяц. ",
    "Пользуюсь каждый день. ",
    "Купил на замену старому. ",
    "Брал для себя. ",
    "Заказывал на подарок. ",
    "Коротко: ",
    "Что могу сказать: ",
    "В целом ",
)


def normalize_review(review):
    """Удаляет повторяющиеся вводные фразы из отзыва."""
    review = review.strip()

    changed = True
    while changed:
        changed = False

        for prefix in PREFIXES:
            if review.startswith(prefix):
                review = review[len(prefix):].strip()
                changed = True

    return review


def generate_card(product):
    """Генерирует карточку товара через локальный vLLM."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    product,
                    ensure_ascii=False
                )
            }
        ],
        temperature=0.2,
        max_tokens=500
    )

    content = response.choices[0].message.content

    return content, response.usage


def main():
    # Загружаем исходные данные
    with open("datasets/dev.jsonl", encoding="utf-8") as file:
        products = [
            json.loads(line)
            for line in file
        ]

    # Выбираем товар P-0039
    product = next(
        p for p in products
        if p["product_id"] == "P-0039"
    )

    original_reviews = product["reviews"]

    # Нормализуем отзывы и подсчитываем частоту тезисов
    normalized_reviews = [
        normalize_review(review)
        for review in original_reviews
    ]

    frequencies = Counter(normalized_reviews)

    # Формируем товар с обработанными отзывами
    prepared_product = {
        **product,
        "reviews": list(frequencies.keys()),
        "review_counts": dict(frequencies)
    }

    # Подсчитываем количество входных токенов
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                prepared_product,
                ensure_ascii=False
            )
        }
    ]

    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )

    # Выводим статистику обработки
    print("Исходных отзывов:", len(original_reviews))
    print("После нормализации:", len(frequencies))
    print("Сокращение:", len(original_reviews) - len(frequencies))
    print("Токенов после обработки:", len(tokens))

    print("\nСохранённые тезисы и частота:")

    for review, count in frequencies.most_common():
        print(f"{count:3} раз | {review}")

    # Генерируем карточку товара
    print("\nГенерация карточки...")

    result, usage = generate_card(prepared_product)

    print("\nРезультат генерации:")
    print(result)

    print("\nИспользование токенов:")
    print(usage)


if __name__ == "__main__":
    main()
