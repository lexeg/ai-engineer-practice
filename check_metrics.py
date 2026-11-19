#!/usr/bin/env python3
"""Проверка карточек по формальным требованиям проекта."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = {"product_id", "description", "pros", "cons", "tags"}

def parse_card_record(record):
    raw = record.get("raw_output") if isinstance(record, dict) else None
    if raw is None:
        return record
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        return json.loads(text)
    raise ValueError("raw_output имеет неподдерживаемый тип")

def count_sentences(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\.(?!\d)|[!?]+", text))

def validate_card(card, expected_product_id):
    errors = []
    if not isinstance(card, dict):
        return ["Карточка должна быть JSON-объектом"]

    actual_fields = set(card)
    if actual_fields != REQUIRED_FIELDS:
        missing = REQUIRED_FIELDS - actual_fields
        extra = actual_fields - REQUIRED_FIELDS
        if missing:
            errors.append("Отсутствуют обязательные поля")
        if extra:
            errors.append("Обнаружены лишние поля")

    if card.get("product_id") != expected_product_id:
        errors.append("product_id не совпадает с исходным")

    description = card.get("description")
    if not isinstance(description, str):
        errors.append("description должен быть строкой")
    else:
        description = description.strip()
        if not 120 <= len(description) <= 700:
            errors.append("Длина description должна быть 120–700 символов")
        if not 2 <= count_sentences(description) <= 5:
            errors.append("description должен содержать 2–5 предложений")

    for field, minimum, maximum in [
        ("pros", 2, 5), ("cons", 1, 3), ("tags", 3, 8)
    ]:
        value = card.get(field)
        if not isinstance(value, list):
            errors.append(f"{field} должен быть массивом")
            continue
        if not minimum <= len(value) <= maximum:
            errors.append(f"{field}: неверное количество элементов")
        if any(not isinstance(x, str) or not x.strip() for x in value):
            errors.append(f"{field}: есть пустые или нестроковые элементы")
        valid_items = [x.strip() for x in value if isinstance(x, str) and x.strip()]
        if len(valid_items) != len(set(valid_items)):
            errors.append(f"{field}: есть дубликаты")
    return errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/predictions.jsonl")
    parser.add_argument("--expected-input", default="datasets/benchmark.jsonl")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                records.append({"_line_error": f"JSONL line {line_no}: {exc}"})

    expected_ids = None
    if args.expected_input and Path(args.expected_input).exists():
        with open(args.expected_input, encoding="utf-8") as f:
            expected_ids = [json.loads(line)["product_id"] for line in f if line.strip()]

    valid = 0
    invalid = 0
    reasons = Counter()
    seen = set()
    duplicate_ids = []
    found_ids = set()

    for record in records:
        if "_line_error" in record:
            invalid += 1
            reasons["ошибка чтения JSONL"] += 1
            continue
        pid = record.get("product_id")
        if pid in seen:
            duplicate_ids.append(pid)
            reasons["повторяющийся product_id"] += 1
        seen.add(pid)
        found_ids.add(pid)
        try:
            card = parse_card_record(record)
            errors = validate_card(card, pid)
        except Exception as exc:
            errors = [f"ошибка разбора JSON: {exc}"]
        if errors:
            invalid += 1
            for error in errors:
                if error.startswith("ошибка разбора"):
                    reasons["некорректный JSON"] += 1
                elif error.startswith("Длина"):
                    reasons["длина description"] += 1
                elif error.startswith("description"):
                    reasons["количество предложений"] += 1
                elif error.startswith("pros"):
                    reasons["поле pros"] += 1
                elif error.startswith("cons"):
                    reasons["поле cons"] += 1
                elif error.startswith("tags"):
                    reasons["поле tags"] += 1
                elif "product_id" in error:
                    reasons["product_id"] += 1
                elif "поля" in error:
                    reasons["структура полей"] += 1
                else:
                    reasons["прочие ошибки"] += 1
        else:
            valid += 1

    missing = sorted(set(expected_ids or []) - found_ids) if expected_ids is not None else []
    if missing:
        invalid += len(missing)
        reasons["отсутствует карточка"] += len(missing)

    denominator = len(expected_ids) if expected_ids is not None else len(records)
    rate = valid / denominator if denominator else 0.0
    result = {
        "input_file": args.input,
        "expected_input": args.expected_input,
        "records": len(records),
        "expected_products": denominator,
        "valid": valid,
        "invalid": denominator - valid,
        "valid_rate": round(rate, 4),
        "errors_by_reason": dict(reasons),
        "duplicate_product_ids": duplicate_ids,
        "missing_product_ids": missing,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
