
import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT_FILE = "outputs/raw_predictions.jsonl"

REQUIRED_FIELDS = {
    "product_id",
    "description",
    "pros",
    "cons",
    "tags"
}

EXPECTED_ARRAY_LENGTHS = {
    "pros": 2,
    "cons": 1,
    "tags": 3
}

CONS_PLACEHOLDERS = {
    "нет",
    "нет недостатков",
    "недостатки отсутствуют",
    "недостатков нет",
    "не обнаружено",
    "не выявлено"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Проверка качества сгенерированных карточек товаров"
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help="Путь к JSONL-файлу с результатами генерации"
    )

    return parser.parse_args()


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


def count_sentences(text):
    """
    Подсчитывает окончания предложений,
    не считая десятичные точки.
    """

    if not isinstance(text, str):
        return 0

    return len(re.findall(r"\.(?!\d)|[!?]+", text))


def validate_card(card, expected_product_id):
    """Проверяет карточку по единому строгому контракту."""

    errors = []

    if not isinstance(card, dict):
        return ["Карточка должна быть JSON-объектом"]

    # Проверка структуры JSON.
    actual_fields = set(card.keys())

    if actual_fields != REQUIRED_FIELDS:
        missing = REQUIRED_FIELDS - actual_fields
        extra = actual_fields - REQUIRED_FIELDS

        if missing:
            errors.append(
                f"Отсутствуют обязательные поля: {sorted(missing)}"
            )

        if extra:
            errors.append(
                f"Обнаружены лишние поля: {sorted(extra)}"
            )

    # Проверка идентификатора товара.
    if card.get("product_id") != expected_product_id:
        errors.append("product_id не совпадает с исходным")

    # Проверка описания.
    description = card.get("description")

    if not isinstance(description, str):
        errors.append("description должен быть строкой")
    else:
        description = description.strip()

        if not 120 <= len(description) <= 700:
            errors.append(
                "Длина description должна быть 120–700 символов"
            )

        sentence_count = count_sentences(description)

        if sentence_count != 3:
            errors.append(
                "description должен содержать ровно 3 предложения"
            )

    # Проверка массивов.
    for field, expected_count in EXPECTED_ARRAY_LENGTHS.items():
        value = card.get(field)

        if not isinstance(value, list):
            errors.append(f"{field} должен быть массивом")
            continue

        if len(value) != expected_count:
            errors.append(
                f"{field} должен содержать ровно "
                f"{expected_count} элементов"
            )

        # Проверка типов и пустых элементов.
        for index, item in enumerate(value):
            if not isinstance(item, str):
                errors.append(
                    f"{field}[{index}] должен быть строкой"
                )
            elif not item.strip():
                errors.append(
                    f"{field}[{index}] не должен быть пустым"
                )

        # Проверка дубликатов.
        valid_items = [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]

        if len(valid_items) != len(set(valid_items)):
            errors.append(f"{field} содержит дубликаты")

        # Проверка заглушек в cons.
        if field == "cons":
            for item in valid_items:
                if item.lower() in CONS_PLACEHOLDERS:
                    errors.append(
                        "cons не должен содержать заглушки "
                        "вместо описания недостатка"
                    )
                    break

    return errors


def load_card(record):
    """Получает карточку из записи JSONL без нормализации."""

    if not isinstance(record, dict):
        raise ValueError("Запись JSONL должна быть объектом")

    raw_output = record.get("raw_output")

    if isinstance(raw_output, dict):
        return raw_output

    if isinstance(raw_output, str):
        return parse_model_json(raw_output)

    raise ValueError(
        "raw_output отсутствует или имеет неверный тип"
    )


def main():
    args = parse_args()
    path = Path(args.input)

    if not path.exists():
        print(f"Файл не найден: {args.input}")
        return

    records = []

    # Читаем JSONL, сохраняя некорректные строки
    # как отдельные невалидные записи.
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                records.append({
                    "_parse_error": (
                        f"Строка {line_number}: {exc}"
                    )
                })

    valid = 0
    invalid = 0

    print(f"Файл: {path}")
    print(f"Загружено карточек: {len(records)}")

    for index, record in enumerate(records, start=1):
        if isinstance(record, dict):
            product_id = record.get("product_id", "UNKNOWN")
        else:
            product_id = "UNKNOWN"

        print(f"\nКарточка №{index} ({product_id})")

        try:
            if not isinstance(record, dict):
                raise ValueError(
                    "Запись JSONL должна быть объектом"
                )

            if "_parse_error" in record:
                raise ValueError(record["_parse_error"])

            card = load_card(record)

            errors = validate_card(
                card,
                product_id
            )

        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            errors = [f"Ошибка чтения карточки: {exc}"]

        if errors:
            invalid += 1

            for error in errors:
                print(f"  ОШИБКА: {error}")
        else:
            valid += 1
            print("  OK")

    total = len(records)

    valid_percent = (
        valid / total * 100
        if total else 0
    )

    print("\n" + "=" * 40)
    print("ИТОГИ ПРОВЕРКИ")
    print("=" * 40)

    print(f"Всего карточек: {total}")
    print(f"Валидных: {valid}")
    print(f"Невалидных: {invalid}")
    print(f"Доля валидных: {valid_percent:.1f}%")


if __name__ == "__main__":
    main()
