#!/usr/bin/env python3
"""Сравнение с OpenRouter. Использует тот же prompt и preprocessing, что run_batch.py."""
import argparse
import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI
import run_batch

API_BASE_URL = "https://openrouter.ai/api/v1"
MAX_API_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/compare_30.jsonl")
    parser.add_argument("--output", default="outputs/api_predictions.jsonl")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=run_batch.MAX_ATTEMPTS)
    return parser.parse_args()

async def request_with_backoff(client, semaphore, **kwargs):
    """429 и 5xx повторяются с экспоненциальной задержкой; 400 не повторяется."""
    total_latency = 0.0
    for retry in range(1, MAX_API_RETRIES + 1):
        try:
            async with semaphore:
                started = time.perf_counter()
                response = await client.chat.completions.create(**kwargs)
                total_latency += time.perf_counter() - started
                return response, total_latency
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            retryable = status == 429 or (isinstance(status, int) and 500 <= status < 600)
            network_error = exc.__class__.__name__ in {"APIConnectionError", "APITimeoutError"}
            if status == 400:
                raise RuntimeError(f"HTTP 400: проверьте параметры запроса: {exc}") from exc
            if not (retryable or network_error) or retry >= MAX_API_RETRIES:
                raise
            await asyncio.sleep(BACKOFF_BASE_SECONDS * (2 ** (retry - 1)))
    raise RuntimeError("Не удалось выполнить API-запрос")

async def process_one(index, original, semaphore, client, model, max_attempts):
    product = run_batch.preprocess_product(original)
    pid = product.get("product_id")
    input_tokens = run_batch.count_input_tokens(product)

    if input_tokens > run_batch.MAX_INPUT_TOKENS:
        return {
            "index": index, "product_id": pid, "status": "INVALID",
            "attempt": 0, "errors": [f"Вход превышает лимит: {input_tokens} токенов"],
            "raw_output": "", "input_tokens": 0, "output_tokens": 0,
            "latency": 0.0, "finish_reasons": []
        }

    previous_output = None
    errors = None
    prompt_tokens_total = 0
    completion_tokens_total = 0
    latency_total = 0.0
    finish_reasons = []
    last_raw = ""

    for attempt in range(1, max_attempts + 1):
        messages = run_batch.build_messages(product, previous_output, errors)
        try:
            response, request_latency = await request_with_backoff(
                client, semaphore,
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=run_batch.MAX_COMPLETION_TOKENS,
            )
            latency_total += request_latency
            choice = response.choices[0]
            last_raw = choice.message.content or ""
            finish_reasons.append(choice.finish_reason)
            if response.usage:
                prompt_tokens_total += getattr(response.usage, "prompt_tokens", 0) or 0
                completion_tokens_total += getattr(response.usage, "completion_tokens", 0) or 0

            card, errors = run_batch.parse_and_validate(last_raw, pid)
            if not errors:
                return {
                    "index": index, "product_id": pid, "status": "OK",
                    "attempt": attempt, "card": card, "errors": [],
                    "raw_output": last_raw, "input_tokens": prompt_tokens_total,
                    "output_tokens": completion_tokens_total,
                    "latency": round(latency_total, 3), "finish_reasons": finish_reasons
                }
            previous_output = card if card is not None else last_raw
        except Exception as exc:
            errors = [str(exc)]
            previous_output = None
            finish_reasons.append("exception")

    return {
        "index": index, "product_id": pid, "status": "INVALID",
        "attempt": max_attempts, "errors": errors or ["Неизвестная ошибка"],
        "raw_output": last_raw, "input_tokens": prompt_tokens_total,
        "output_tokens": completion_tokens_total,
        "latency": round(latency_total, 3), "finish_reasons": finish_reasons
    }

async def main():
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")
    if not api_key:
        raise RuntimeError("Задайте OPENROUTER_API_KEY в переменной окружения")
    if not model:
        raise RuntimeError("Задайте OPENROUTER_MODEL в переменной окружения")
    if args.concurrency < 1:
        raise ValueError("--concurrency должен быть >= 1")

    products = run_batch.read_jsonl(args.input)
    if not products:
        raise ValueError(f"Входной файл пуст: {args.input}")

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=api_key, timeout=120.0)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    try:
        tasks = [
            process_one(i, p, semaphore, client, model, args.max_attempts)
            for i, p in enumerate(products)
        ]
        results = await asyncio.gather(*tasks)
    finally:
        await client.close()
        await run_batch.client.close()
    elapsed = time.perf_counter() - started

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            record = {
                "product_id": r["product_id"],
                "raw_output": json.dumps(r["card"], ensure_ascii=False) if r.get("card") is not None else r.get("raw_output", ""),
                "status": r["status"], "attempt": r["attempt"],
                "errors": r.get("errors", []),
                "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                "latency": r["latency"], "finish_reasons": r["finish_reasons"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(results)
    valid = sum(r["status"] == "OK" for r in results)
    input_tokens = sum(r["input_tokens"] for r in results)
    output_tokens = sum(r["output_tokens"] for r in results)
    cost = ((input_tokens * 20) + (output_tokens * 60)) / 1_000_000 / total * 1000
    report = {
        "model": model, "n": total, "valid": valid, "invalid": total-valid,
        "valid_rate": valid/total if total else 0,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "elapsed_sec": round(elapsed, 3),
        "throughput": total/elapsed if elapsed else None,
        "avg_request_latency_sec": round(sum(r["latency"] for r in results)/total, 3) if total else None,
        "cost_per_1000_rub": round(cost, 3),
        "concurrency": args.concurrency,
        "temperature": 0.2,
        "max_tokens": run_batch.MAX_COMPLETION_TOKENS,
        "prompt_source": "run_batch.SYSTEM_PROMPT via run_batch.build_messages",
        "api_retry_policy": {"max_retries": MAX_API_RETRIES, "retry_statuses": [429, "5xx"], "backoff": "1, 2, 4 seconds"},
    }
    (out.parent / "api_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
