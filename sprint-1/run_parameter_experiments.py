import asyncio
import csv
import itertools
import json
import re
import time
from pathlib import Path

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from run_batch import (
    MODEL,
    BASE_URL,
    API_KEY,
    SYSTEM_PROMPT,
    preprocess_product,
    count_input_tokens,
)
from check_metrics import validate_card, parse_model_json


INPUT_FILE = "datasets/dev.jsonl"
RESULTS_FILE = "outputs/parameter_experiments.jsonl"
REPORT_FILE = "outputs/parameter_experiments.csv"

PRODUCT_COUNT = 20
CONCURRENCY = 1
MAX_INPUT_TOKENS = 1500

TEMPERATURES = [0.0, 0.3, 0.7]
TOP_P_VALUES = [0.9, 1.0]
TOP_K_VALUES = [20, 50, -1]
REPETITION_PENALTIES = [1.0, 1.1, 1.2]
MAX_TOKENS_VALUES = [250, 350, 500]

tokenizer = AutoTokenizer.from_pretrained(MODEL)

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=180.0,
)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def append_jsonl(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed(path):
    completed = set()

    if not Path(path).exists():
        return completed

    for record in read_jsonl(path):
        completed.add((
            record["config_id"],
            record["product_id"],
        ))

    return completed


def count_sentences(text):
    return len(re.findall(r"\.(?!\d)|[!?]+", text))


def configuration_grid():
    values = itertools.product(
        TEMPERATURES,
        TOP_P_VALUES,
        TOP_K_VALUES,
        REPETITION_PENALTIES,
        MAX_TOKENS_VALUES,
    )

    for index, (temperature, top_p, top_k, repetition, max_tokens) in enumerate(values):
        yield {
            "config_id": index,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition,
            "max_tokens": max_tokens,
        }


def build_messages(product):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(product, ensure_ascii=False),
        },
    ]


async def process_product(product, config, semaphore):
    product = preprocess_product(product)
    product_id = product.get("product_id")

    input_tokens = count_input_tokens(product)

    if input_tokens > MAX_INPUT_TOKENS:
        return {
            "config_id": config["config_id"],
            "product_id": product_id,
            "status": "INVALID",
            "errors": [f"Input tokens limit exceeded: {input_tokens}"],
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "description_length": 0,
        }

    messages = build_messages(product)
    started = time.perf_counter()

    try:
        async with semaphore:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=config["temperature"],
                top_p=config["top_p"],
                max_tokens=config["max_tokens"],
                extra_body={
                    "top_k": config["top_k"],
                    "repetition_penalty": config["repetition_penalty"],
                },
            )

        elapsed = time.perf_counter() - started
        choice = response.choices[0]
        raw_output = choice.message.content or ""

        output_tokens = (
            response.usage.completion_tokens
            if response.usage
            else len(tokenizer.encode(raw_output))
        )

        try:
            card = parse_model_json(raw_output)
            errors = validate_card(card, product_id)
        except (ValueError, json.JSONDecodeError, TypeError) as exc:
            card = None
            errors = [f"JSON parsing error: {exc}"]

        description = (
            card.get("description", "")
            if isinstance(card, dict)
            else ""
        )

        return {
            "config_id": config["config_id"],
            "product_id": product_id,
            "status": "OK" if not errors else "INVALID",
            "errors": errors,
            "raw_output": raw_output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "description_length": len(description)
            if isinstance(description, str) else 0,
            "sentence_count": count_sentences(description)
            if isinstance(description, str) else 0,
            "elapsed_sec": round(elapsed, 3),
            "finish_reason": choice.finish_reason,
        }

    except Exception as exc:
        return {
            "config_id": config["config_id"],
            "product_id": product_id,
            "status": "ERROR",
            "errors": [str(exc)],
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "description_length": 0,
        }


async def run_configuration(products, config, completed):
    semaphore = asyncio.Semaphore(CONCURRENCY)

    pending = [
        product for product in products
        if (
            config["config_id"],
            product.get("product_id"),
        ) not in completed
    ]

    if not pending:
        return

    print(
        f"\nКонфигурация {config['config_id'] + 1}/162: "
        f"{config}",
        flush=True,
    )

    tasks = [
        process_product(product, config, semaphore)
        for product in pending
    ]

    results = await asyncio.gather(*tasks)

    for result in results:
        append_jsonl(RESULTS_FILE, result)
        completed.add((
            config["config_id"],
            result["product_id"],
        ))

    valid = sum(result["status"] == "OK" for result in results)

    print(
        f"Результат: {valid}/{len(results)} валидных",
        flush=True,
    )


def build_report():
    records = read_jsonl(RESULTS_FILE)
    grouped = {}

    for record in records:
        grouped.setdefault(record["config_id"], []).append(record)

    configs = {
        config["config_id"]: config
        for config in configuration_grid()
    }

    Path(REPORT_FILE).parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "config_id",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_tokens",
        "total",
        "valid",
        "invalid",
        "valid_rate",
        "avg_description_length",
        "avg_output_tokens",
        "total_output_tokens",
        "avg_latency_sec",
    ]

    with open(REPORT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for config_id, items in sorted(grouped.items()):
            config = configs[config_id]
            successful = [
                item for item in items
                if item["status"] == "OK"
            ]

            total = len(items)
            valid = len(successful)

            writer.writerow({
                **config,
                "total": total,
                "valid": valid,
                "invalid": sum(
                    item["status"] != "OK" for item in items
                ),
                "valid_rate": round(valid / total, 4) if total else 0,
                "avg_description_length": round(
                    sum(item.get("description_length", 0) for item in items)
                    / total, 2
                ) if total else 0,
                "avg_output_tokens": round(
                    sum(item.get("output_tokens", 0) for item in items)
                    / total, 2
                ) if total else 0,
                "total_output_tokens": sum(
                    item.get("output_tokens", 0) for item in items
                ),
                "avg_latency_sec": round(
                    sum(item.get("elapsed_sec", 0) for item in items)
                    / total, 3
                ) if total else 0,
            })


async def main():
    products = read_jsonl(INPUT_FILE)[:PRODUCT_COUNT]

    if len(products) != PRODUCT_COUNT:
        raise ValueError(
            f"Ожидалось {PRODUCT_COUNT} товаров, найдено {len(products)}"
        )

    completed = load_completed(RESULTS_FILE)
    configs = list(configuration_grid())

    print(f"Товаров: {len(products)}")
    print(f"Конфигураций: {len(configs)}")
    print(f"Всего запросов: {len(products) * len(configs)}")
    print(f"Уже выполнено: {len(completed)}")

    try:
        for config in configs:
            await run_configuration(products, config, completed)
    finally:
        await client.close()

    build_report()

    print("\nЭксперимент завершён.")
    print(f"Результаты: {RESULTS_FILE}")
    print(f"Отчёт: {REPORT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
