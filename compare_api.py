#!/usr/bin/env python3
"""Сравнение с OpenRouter с диагностикой генерации и повторными попытками."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from openai import AsyncOpenAI
import run_batch


API_BASE_URL = "https://openrouter.ai/api/v1"

MAX_API_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.0

# Лимит для генерации карточки.
API_MAX_COMPLETION_TOKENS = 2000


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="datasets/compare_30.jsonl"
    )

    parser.add_argument(
        "--output",
        default="outputs/api_predictions_30.jsonl"
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=4
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=run_batch.MAX_ATTEMPTS
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=API_MAX_COMPLETION_TOKENS
    )

    return parser.parse_args()


def get_message_content(message):
    """Безопасно извлекает текст ответа."""

    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            text = getattr(item, "text", None)

            if text:
                parts.append(text)

        return "\n".join(parts)

    return ""


def get_reasoning(message):
    """Извлекает reasoning, если провайдер его предоставляет."""

    reasoning = getattr(message, "reasoning", None)

    if reasoning is None:
        reasoning = getattr(message, "reasoning_content", None)

    if isinstance(reasoning, str):
        return reasoning

    return ""


async def request_with_backoff(client, semaphore, **kwargs):
    """Первичная попытка + до трех повторов при сетевых ошибках."""

    total_latency = 0.0

    for attempt in range(MAX_API_RETRIES + 1):
        try:
            async with semaphore:
                started = time.perf_counter()

                response = await client.chat.completions.create(
                    **kwargs
                )

                total_latency += time.perf_counter() - started

            return response, total_latency

        except Exception as exc:
            status = getattr(exc, "status_code", None)

            retryable = (
                status == 429
                or (
                    isinstance(status, int)
                    and 500 <= status < 600
                )
            )

            network_error = exc.__class__.__name__ in {
                "APIConnectionError",
                "APITimeoutError"
            }

            if status == 400:
                raise RuntimeError(
                    f"HTTP 400: проверьте параметры запроса: {exc}"
                ) from exc

            if not (retryable or network_error):
                raise

            if attempt >= MAX_API_RETRIES:
                raise

            delay = BACKOFF_BASE_SECONDS * (2 ** attempt)
            await asyncio.sleep(delay)

    raise RuntimeError("Не удалось выполнить API-запрос")


async def process_one(
    index,
    original,
    semaphore,
    client,
    model,
    max_attempts,
    max_tokens
):
    product = run_batch.preprocess_product(original)
    pid = product.get("product_id")

    input_tokens = run_batch.count_input_tokens(product)

    if input_tokens > run_batch.MAX_INPUT_TOKENS:
        return {
            "index": index,
            "product_id": pid,
            "status": "INVALID",
            "attempt": 0,
            "errors": [
                f"Вход превышает лимит: {input_tokens} токенов"
            ],
            "raw_output": "",
            "reasoning": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency": 0.0,
            "finish_reasons": [],
            "attempt_details": []
        }

    previous_output = None
    errors = None

    prompt_tokens_total = 0
    completion_tokens_total = 0
    latency_total = 0.0

    finish_reasons = []
    attempt_details = []

    last_raw = ""
    last_reasoning = ""

    for attempt in range(1, max_attempts + 1):

        messages = run_batch.build_messages(
            product,
            previous_output,
            errors
        )

        try:
            response, request_latency = await request_with_backoff(
                client,
                semaphore,
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens
            )

            latency_total += request_latency

            if not response.choices:
                raise RuntimeError(
                    "API вернул ответ без choices"
                )

            choice = response.choices[0]
            message = choice.message

            last_raw = get_message_content(message)
            last_reasoning = get_reasoning(message)

            finish_reason = choice.finish_reason or "unknown"
            finish_reasons.append(finish_reason)

            usage = response.usage

            prompt_tokens = (
                getattr(usage, "prompt_tokens", 0) or 0
                if usage else 0
            )

            completion_tokens = (
                getattr(usage, "completion_tokens", 0) or 0
                if usage else 0
            )

            prompt_tokens_total += prompt_tokens
            completion_tokens_total += completion_tokens

            detail = {
                "attempt": attempt,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency": round(request_latency, 3),
                "content_length": len(last_raw),
                "reasoning_length": len(last_reasoning)
            }

            attempt_details.append(detail)

            # Не пытаемся разбирать заведомо оборванный ответ.
            if finish_reason == "length":
                errors = [
                    f"Ответ обрезан: finish_reason=length, "
                    f"лимит={max_tokens} токенов"
                ]

                previous_output = None
                continue

            if not last_raw.strip():
                errors = [
                    "API вернул пустой content"
                ]

                previous_output = None
                continue

            card, validation_errors = run_batch.parse_and_validate(
                last_raw,
                pid
            )

            if not validation_errors:
                return {
                    "index": index,
                    "product_id": pid,
                    "status": "OK",
                    "attempt": attempt,
                    "card": card,
                    "errors": [],
                    "raw_output": last_raw,
                    "reasoning": last_reasoning,
                    "input_tokens": prompt_tokens_total,
                    "output_tokens": completion_tokens_total,
                    "latency": round(latency_total, 3),
                    "finish_reasons": finish_reasons,
                    "attempt_details": attempt_details
                }

            errors = validation_errors
            previous_output = (
                card if card is not None else last_raw
            )

        except Exception as exc:
            errors = [f"{type(exc).__name__}: {exc}"]
            previous_output = None

            finish_reasons.append("exception")

            attempt_details.append({
                "attempt": attempt,
                "finish_reason": "exception",
                "error": str(exc)
            })

    return {
        "index": index,
        "product_id": pid,
        "status": "INVALID",
        "attempt": max_attempts,
        "errors": errors or ["Неизвестная ошибка"],
        "raw_output": last_raw,
        "reasoning": last_reasoning,
        "input_tokens": prompt_tokens_total,
        "output_tokens": completion_tokens_total,
        "latency": round(latency_total, 3),
        "finish_reasons": finish_reasons,
        "attempt_details": attempt_details
    }


async def main():
    args = parse_args()

    if args.concurrency < 1:
        raise ValueError("--concurrency должен быть >= 1")

    if args.max_attempts < 1:
        raise ValueError("--max-attempts должен быть >= 1")

    if args.max_tokens < 1:
        raise ValueError("--max-tokens должен быть >= 1")

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")

    if not api_key:
        raise RuntimeError(
            "Задайте OPENROUTER_API_KEY в переменной окружения"
        )

    if not model:
        raise RuntimeError(
            "Задайте OPENROUTER_MODEL в переменной окружения"
        )

    products = run_batch.read_jsonl(args.input)

    if not products:
        raise ValueError(
            f"Входной файл пуст: {args.input}"
        )

    client = AsyncOpenAI(
        base_url=API_BASE_URL,
        api_key=api_key,
        timeout=180.0
    )

    semaphore = asyncio.Semaphore(args.concurrency)

    started = time.perf_counter()

    try:
        tasks = [
            process_one(
                i,
                product,
                semaphore,
                client,
                model,
                args.max_attempts,
                args.max_tokens
            )
            for i, product in enumerate(products)
        ]

        results = await asyncio.gather(*tasks)

    finally:
        await client.close()
        await run_batch.client.close()

    elapsed = time.perf_counter() - started

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as file:
        for result in results:

            record = {
                "product_id": result["product_id"],
                "raw_output": (
                    json.dumps(
                        result["card"],
                        ensure_ascii=False
                    )
                    if result.get("card") is not None
                    else result.get("raw_output", "")
                ),
                "reasoning": result.get("reasoning", ""),
                "status": result["status"],
                "attempt": result["attempt"],
                "errors": result.get("errors", []),
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "latency": result["latency"],
                "finish_reasons": result["finish_reasons"],
                "attempt_details": result.get("attempt_details", [])
            }

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    total = len(results)

    valid = sum(
        result["status"] == "OK"
        for result in results
    )

    invalid = total - valid

    input_tokens = sum(
        result["input_tokens"]
        for result in results
    )

    output_tokens = sum(
        result["output_tokens"]
        for result in results
    )

    cost = (
        (input_tokens * 20) + (output_tokens * 60)
    ) / 1_000_000 / total * 1000 if total else 0

    finish_reason_counts = {}

    for result in results:
        for reason in result["finish_reasons"]:
            finish_reason_counts[reason] = (
                finish_reason_counts.get(reason, 0) + 1
            )

    report = {
        "model": model,
        "n": total,
        "valid": valid,
        "invalid": invalid,
        "valid_rate": valid / total if total else 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_sec": round(elapsed, 3),
        "throughput": total / elapsed if elapsed else None,
        "avg_latency_per_card_sec": (
            round(
                sum(r["latency"] for r in results) / total,
                3
            )
            if total else None
        ),
        "cost_per_1000_rub": round(cost, 3),
        "concurrency": args.concurrency,
        "temperature": 0.2,
        "max_tokens": args.max_tokens,
        "prompt_source": (
            "run_batch.SYSTEM_PROMPT via run_batch.build_messages"
        ),
        "finish_reason_counts": finish_reason_counts,
        "api_retry_policy": {
            "max_retries": MAX_API_RETRIES,
            "retry_statuses": [429, "5xx"],
            "backoff": "1, 2, 4 seconds"
        }
    }

    report_path = out.parent / "api_report.json"

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
