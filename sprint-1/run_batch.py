
import argparse
import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from check_metrics import validate_card


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"

MAX_ATTEMPTS = 2
MAX_INPUT_TOKENS = 1500
MAX_COMPLETION_TOKENS = 800

TEMPERATURE = 0.2
TOP_P = 1.0
TOP_K = -1
REPETITION_PENALTY = 1.0

CONCURRENCY = 1

INPUT_FILE = "datasets/dev.jsonl"
OUTPUT_FILE = None

tokenizer = AutoTokenizer.from_pretrained(MODEL)

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=180.0
)


SYSTEM_PROMPT = """
Ты — редактор карточек товаров маркетплейса.

На основании названия, категории, атрибутов и отзывов
составь качественную карточку товара.

Верни только JSON-объект без Markdown и пояснений.

JSON должен содержать ровно 5 полей:
- product_id
- description
- pros
- cons
- tags

ТРЕБОВАНИЯ:

product_id:
- Точное значение из входных данных.

description:
- От 120 до 700 символов.
- Желательная длина: 150–250 символов.
- Ровно 3 законченных предложения.

Используй следующую структуру:

ПРЕДЛОЖЕНИЕ 1:
Название товара, его назначение и основные характеристики.

ПРЕДЛОЖЕНИЕ 2:
Особенности товара и преимущества, подтверждённые отзывами.

ПРЕДЛОЖЕНИЕ 3:
Дополнительная полезная информация, впечатления покупателей
или подтверждённые недостатки.

ВАЖНЫЕ ПРАВИЛА ДЛЯ DESCRIPTION:
- Каждое предложение должно выражать законченную мысль.
- Каждое предложение заканчивай точкой.
- Не пиши описание как список характеристик.
- Не перечисляй характеристики через запятые.
- Не объединяй несколько предложений в одно.
- Не используй обрывки фраз.
- Не оставляй незаконченные слова.
- Не начинай описание только с отзывов.
- Обязательно используй название товара в первом предложении.
- Не добавляй выдуманные характеристики.
- Атрибуты товара являются главным источником технических характеристик.
- Если отзыв противоречит атрибуту, всегда используй значение атрибута.
- Не переноси характеристики из отзывов, если они противоречат атрибутам.
- Если атрибут явно содержит значение "нет", не утверждай обратное.
- Отзывы используй для описания пользовательского опыта,
  ощущений и реальных преимуществ или недостатков.
- Если информации недостаточно, используй только подтверждённые факты.
- Не повторяй одну мысль разными словами.

pros:
- Массив ровно из 2 непустых строк.
- Только два наиболее важных преимущества.
- Каждая строка не более 10 слов.
- Только подтверждённые данными преимущества.

cons:
- Массив ровно из 1 непустой строки.
- Только один наиболее важный подтверждённый недостаток
  или ограничение товара.
- Не выдумывай недостатки.
- pros должны содержать только положительные свойства товара.
- cons должны содержать только отрицательные свойства или ограничения.
- Не переноси отрицательные отзывы в pros.
- Не переноси положительные отзывы в cons.
- Не путай упоминание характеристики с оценкой этой характеристики.
- Если отзывы противоречат друг другу, не делай категоричных выводов.

tags:
- Массив ровно из 3 непустых строк.
- Только три наиболее важные характеристики товара.
- Каждая метка должна быть краткой.

ОБЩИЕ ПРАВИЛА:
- Не используй дубликаты.
- Не добавляй дополнительные поля.
- Не копируй отзывы целиком.
- Не выдумывай отсутствующие факты.
- Не изменяй значения характеристик и единицы измерения.
- Не приписывай товару отсутствующие функции и комплектующие.
- Не делай выводы, которые не подтверждены атрибутами или отзывами.
- Не добавляй рекламные обещания без подтверждения данными.

ПЕРЕД ОТПРАВКОЙ ПРОВЕРЬ:
1. JSON содержит ровно 5 полей.
2. description содержит от 120 до 700 символов.
3. description состоит ровно из 3 предложений.
4. Каждое предложение заканчивается точкой.
5. pros содержит ровно 2 элемента.
6. cons содержит ровно 1 элемент.
7. tags содержит ровно 3 элемента.

Верни только корректный JSON.
"""


PREFIXES = (
    "Пользуюсь пару недель. ",
    "Пользуюсь месяц. ",
    "Пользуюсь каждый день. ",
    "Купил на замену старому. ",
    "Брал для себя. ",
    "Заказывал на подарок. ",
    "Коротко: ",
    "Что могу сказать: ",
    "В целом "
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark генерации карточек товаров через vLLM"
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Количество параллельных запросов"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="datasets/dev.jsonl",
        help="Путь к входному JSONL-файлу"
    )

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=800)

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь к выходному JSONL-файлу"
    )

    args = parser.parse_args()

    if args.concurrency < 1:
        parser.error("--concurrency должен быть не меньше 1")

    return args


def normalize_review(review):
    """Удаляет типовые вводные фразы из отзывов."""

    if not isinstance(review, str):
        return ""

    review = review.strip()

    for prefix in PREFIXES:
        if review.startswith(prefix):
            review = review[len(prefix):]
            break

    return review.strip()


def filter_conflicting_reviews(reviews):
    """Удаляет отзывы с противоречивыми характеристиками."""

    filtered = []

    for review in reviews:
        text = review.lower()

        if "водонепроницаем" in text and "не водонепроницаем" in text:
            continue

        if "есть отделение для ноутбука" in text:
            continue

        if "нет отделения для ноутбука" in text:
            continue

        filtered.append(review)

    return filtered


def preprocess_product(product):
    """Подготавливает товар и подбирает отзывы под лимит токенов."""

    reviews = product.get("reviews", [])

    normalized_reviews = [
        normalize_review(review)
        for review in reviews
    ]

    normalized_reviews = [
        review
        for review in normalized_reviews
        if review
    ]

    counts = Counter(normalized_reviews)

    unique_reviews = list(counts.keys())
    unique_reviews = filter_conflicting_reviews(unique_reviews)

    # Сначала рассматриваем не более семи уникальных отзывов.
    selected_reviews = unique_reviews[:7]

    result = dict(product)

    # Удаляем отзывы с конца, пока запрос не поместится в лимит.
    while selected_reviews:
        result["reviews"] = selected_reviews
        result["review_counts"] = {
            review: counts[review]
            for review in selected_reviews
        }

        if count_input_tokens(result) <= MAX_INPUT_TOKENS:
            return result

        selected_reviews.pop()

    # Проверяем товар без отзывов.
    result["reviews"] = []
    result["review_counts"] = {}

    return result


def count_input_tokens(product):
    """Подсчитывает количество токенов входного сообщения."""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": json.dumps(product, ensure_ascii=False)
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return len(tokenizer.encode(text))


def normalize_card(card):
    """Нормализует структуру карточки."""

    if not isinstance(card, dict):
        return card

    if isinstance(card.get("cons"), str):
        card["cons"] = [card["cons"]]

    limits = {
        "pros": 2,
        "cons": 1,
        "tags": 3
    }

    for field, limit in limits.items():
        value = card.get(field)

        if not isinstance(value, list):
            continue

        items = [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]

        unique_items = list(dict.fromkeys(items))

        card[field] = unique_items[:limit]

    description = card.get("description")

    if isinstance(description, str):
        sentences = re.findall(
            r".+?(?:[.!?]+(?=\s|$)|$)",
            description.strip()
        )

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        if len(sentences) > 3:
            card["description"] = " ".join(sentences[:3])

    return card


def parse_model_json(raw_output):
    """Извлекает JSON из обычного ответа или Markdown-блока."""

    if not isinstance(raw_output, str):
        raise ValueError("Ответ модели должен быть строкой")

    text = raw_output.strip()

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text
    )

    return json.loads(text)


def parse_and_validate(raw_output, product_id):
    """Разбирает JSON и проверяет карточку."""

    try:
        card = parse_model_json(raw_output)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, [f"Некорректный JSON: {exc}"]

    card = normalize_card(card)

    errors = validate_card(card, product_id)

    return card, errors


def build_messages(product, previous_output=None, errors=None):
    """Формирует сообщения для первой или повторной генерации."""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": json.dumps(product, ensure_ascii=False)
        }
    ]

    if previous_output is not None:
        correction_prompt = {
            "previous_output": previous_output,
            "validation_errors": errors,
            "instruction": (
                "Полностью исправь карточку товара. "
                "Используй исходные данные и ошибки проверки.\n\n"

                "Верни JSON ровно с 5 полями: "
                "product_id, description, pros, cons, tags.\n\n"

                "DESCRIPTION — ОБЯЗАТЕЛЬНО ПЕРЕПИШИ ПОЛНОСТЬЮ:\n"
                "- Длина строго от 120 до 700 символов.\n"
                "- Ровно 3 законченных предложения.\n"
                "- Целевая длина: 150–250 символов.\n"
                "- Первое предложение начинается с точного названия товара "
                "и описывает его назначение и основные характеристики.\n"
                "- Второе предложение описывает подтверждённые преимущества "
                "и пользовательский опыт.\n"
                "- Третье предложение содержит дополнительную полезную "
                "информацию или подтверждённое ограничение.\n"
                "- Каждое предложение заканчивается точкой.\n"
                "- Между предложениями должны быть пробелы.\n"
                "- Не перечисляй характеристики через запятые. "
                "Используй связные законченные фразы.\n"
                "- Не объединяй описание в одно предложение.\n"
                "- Не используй обрывки фраз и незаконченные слова.\n"
                "- Не копируй ошибочное описание без исправлений.\n"
                "- Перед отправкой пересчитай предложения и символы.\n\n"

                "МАССИВЫ:\n"
                "- pros: ровно 2 непустые строки.\n"
                "- cons: ровно 1 непустая строка.\n"
                "- cons должен содержать только подтверждённый недостаток или ограничение товара.\n"
                "- Не используй заглушки: 'нет', 'нет недостатков', 'недостатки отсутствуют', 'не обнаружено'.\n"
                "- Не придумывай недостатки, которых нет в исходных данных.\n"
                "- Если недостатки не подтверждены, используй нейтральную формулировку об отсутствии выявленных недостатков, если это допускает схема задания.\n"
                "- tags: ровно 3 непустые строки.\n"
                "- Не добавляй дополнительные элементы.\n"
                "- Не используй дубликаты.\n"


                "Не придумывай факты. Исправь все ошибки проверки. "
                "Атрибуты являются главным источником технических характеристик. "
                "При противоречии отзыва и атрибута используй атрибут. "
                "pros должны содержать только положительные свойства, "
                "cons — только отрицательные свойства или ограничения. "
                "Не переноси положительные отзывы в cons, "
                "а отрицательные — в pros. "
                "Не добавляй неподтверждённые характеристики. "
                "Бренд используй только из названия или атрибутов товара. "
                "Не используй бренды других товаров. "
                "Если атрибут содержит значение 'нет', "
                "не утверждай наличие этой функции. "
                "Не добавляй характеристики из противоречивых отзывов. "
                "Пиши только на русском языке. "
                "Не используй случайные иностранные слова и символы. "
                "Перед отправкой проверь все поля JSON. "
                "Верни только корректный JSON без Markdown."
            )
        }

        messages.append({
            "role": "user",
            "content": json.dumps(
                correction_prompt,
                ensure_ascii=False
            )
        })

    return messages


async def process_product(index, original_product, semaphore):
    """Обрабатывает одну карточку с возможностью Retry."""

    product = preprocess_product(original_product)

    product_id = product.get("product_id")

    input_tokens = count_input_tokens(product)

    if input_tokens > MAX_INPUT_TOKENS:
        return {
            "index": index,
            "product_id": product_id,
            "status": "INVALID",
            "attempt": 0,
            "errors": [
                f"Превышен лимит входных токенов: {input_tokens}"
            ],
            "raw_output": "",
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "finish_reasons": []
        }

    previous_output = None
    errors = None

    total_output_tokens = 0
    finish_reasons = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            messages = build_messages(
                product,
                previous_output,
                errors
            )

            request_max_tokens = MAX_COMPLETION_TOKENS

            if (
                attempt > 1
                and finish_reasons
                and finish_reasons[-1] == "length"
            ):
                request_max_tokens = max(
                    MAX_COMPLETION_TOKENS,
                    500
                )

            async with semaphore:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=request_max_tokens,
                    extra_body={
                        "top_k": TOP_K,
                        "repetition_penalty": REPETITION_PENALTY
                    }
                )

            choice = response.choices[0]

            raw_output = choice.message.content or ""

            finish_reasons.append(choice.finish_reason)

            total_output_tokens += (
                response.usage.completion_tokens
                if response.usage
                else len(tokenizer.encode(raw_output))
            )

            card, errors = parse_and_validate(
                raw_output,
                product_id
            )

            if not errors:
                return {
                    "index": index,
                    "product_id": product_id,
                    "status": "OK",
                    "attempt": attempt,
                    "card": card,
                    "raw_output": raw_output,
                    "input_tokens": input_tokens,
                    "output_tokens": total_output_tokens,
                    "finish_reasons": finish_reasons
                }

            previous_output = (
                card if card is not None else raw_output
            )

        except Exception as exc:
            finish_reasons.append("exception")
            errors = [f"Ошибка генерации: {str(exc)}"]
            previous_output = None

    return {
        "index": index,
        "product_id": product_id,
        "status": "INVALID",
        "attempt": MAX_ATTEMPTS,
        "errors": errors,
        "raw_output": previous_output,
        "input_tokens": input_tokens,
        "output_tokens": total_output_tokens,
        "finish_reasons": finish_reasons
    }


async def run_concurrent(products):
    """Запускает обработку карточек с ограничением concurrency."""

    semaphore = asyncio.Semaphore(CONCURRENCY)

    tasks = [
        process_product(index, product, semaphore)
        for index, product in enumerate(products)
    ]

    results = []
    completed = 0

    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)

        completed += 1

        print(
            f"[{completed}/{len(products)}] "
            f"{result['product_id']}: "
            f"{result['status']} "
            f"(attempt={result['attempt']}, "
            f"finish_reason={result['finish_reasons']})",
            flush=True
        )

    results.sort(key=lambda item: item["index"])

    return results


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def write_jsonl(path, records):
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


async def main():
    global CONCURRENCY
    global TEMPERATURE, TOP_P, TOP_K
    global REPETITION_PENALTY, MAX_COMPLETION_TOKENS

    args = parse_args()

    CONCURRENCY = args.concurrency
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    TOP_K = args.top_k
    REPETITION_PENALTY = args.repetition_penalty
    MAX_COMPLETION_TOKENS = args.max_tokens

    input_file = args.input

    if args.output:
        output_file = args.output
    else:
        input_path = Path(input_file)
        output_file = (
            f"outputs/{input_path.stem}_c{CONCURRENCY}.jsonl"
        )

    products = read_jsonl(input_file)

    if not products:
        raise ValueError(f"Входной файл пуст: {input_file}")

    print("=" * 60)
    print("BENCHMARK ГЕНЕРАЦИИ КАРТОЧЕК")
    print("=" * 60)
    print(f"Модель: {MODEL}")
    print(f"Входной файл: {input_file}")
    print(f"Загружено товаров: {len(products)}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Max attempts: {MAX_ATTEMPTS}")
    print(f"Max input tokens: {MAX_INPUT_TOKENS}")
    print(f"Max completion tokens: {MAX_COMPLETION_TOKENS}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Top P: {TOP_P}")
    print(f"Top K: {TOP_K}")
    print(f"Repetition penalty: {REPETITION_PENALTY}")
    print(f"Выходной файл: {output_file}")
    print("=" * 60)

    start_time = time.perf_counter()

    try:
        results = await run_concurrent(products)
    finally:
        await client.close()

    elapsed = time.perf_counter() - start_time

    records = []

    for result in results:
        records.append({
            "product_id": result["product_id"],
            "raw_output": (
                json.dumps(result["card"], ensure_ascii=False)
                if result.get("card") is not None
                else result.get("raw_output", "")
            ),
            "status": result["status"],
            "attempt": result["attempt"],
            "errors": result.get("errors", []),
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "finish_reasons": result.get("finish_reasons", [])
        })

    write_jsonl(output_file, records)

    successful_first = sum(
        1 for item in results
        if item["status"] == "OK" and item["attempt"] == 1
    )

    successful_retry = sum(
        1 for item in results
        if item["status"] == "OK" and item["attempt"] == 2
    )

    invalid = sum(
        1 for item in results
        if item["status"] == "INVALID"
    )

    total = len(results)

    input_tokens = sum(
        item["input_tokens"]
        for item in results
    )

    output_tokens = sum(
        item["output_tokens"]
        for item in results
    )

    finish_reason_counts = Counter(
        reason
        for item in results
        for reason in item.get("finish_reasons", [])
    )

    valid_percent = (
        (successful_first + successful_retry) / total * 100
        if total else 0
    )

    print("\n" + "=" * 60)
    print("ИТОГИ ГЕНЕРАЦИИ")
    print("=" * 60)

    print(f"Всего карточек: {total}")
    print(f"Успешно с первой попытки: {successful_first}")
    print(f"Успешно после Retry: {successful_retry}")
    print(f"Невалидных: {invalid}")
    print(f"Доля валидных: {valid_percent:.1f}%")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Время выполнения: {elapsed:.1f} сек.")

    print(
        f"Скорость: {total / elapsed:.2f} карточек/сек."
        if elapsed else "Скорость: 0"
    )

    print(f"Всего входных токенов API: {input_tokens}")
    print(f"Всего выходных токенов API: {output_tokens}")

    print("\nПричины завершения генерации:")

    for reason, count in finish_reason_counts.items():
        print(f"  {reason}: {count}")

    print(f"\nРезультаты сохранены в {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
