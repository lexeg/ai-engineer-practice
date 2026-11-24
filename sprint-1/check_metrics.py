import argparse
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_INPUT_FILE = "outputs/raw_predictions.jsonl"

REQUIRED_FIELDS = {
    "product_id",
    "description",
    "pros",
    "cons",
    "tags",
}

ARRAY_LENGTH_RANGES = {
    "pros": (2, 5),
    "cons": (1, 3),
    "tags": (3, 8),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Проверка качества сгенерированных карточек товаров"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help="Путь к JSONL-файлу с результатами генерации",
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
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)

    return json.loads(text)


def count_sentences(text):
    """
    Подсчитывает окончания предложений,
    не считая десятичные точки внутри чисел.
    """

    if not isinstance(text, str):
        return 0

    return len(re.findall(r"\.(?!\d)|[!?]+", text))


def validate_card(card, expected_product_id):
    """
    Проверяет карточку по обязательным правилам формата задания.

    Дополнительные поля, повторяющиеся элементы и заглушки в cons
    не влияют на основную метрику валидности.
    """

    errors = []

    if not isinstance(card, dict):
        return [("card_type", "Карточка должна быть JSON-объектом")]

    # Обязательные поля должны присутствовать, дополнительные разрешены.
    missing = REQUIRED_FIELDS - set(card.keys())
    if missing:
        errors.append((
            "missing_fields",
            f"Отсутствуют обязательные поля: {sorted(missing)}",
        ))

    # Проверка идентификатора товара.
    if card.get("product_id") != expected_product_id:
        errors.append((
            "product_id_mismatch",
            "product_id не совпадает с исходным",
        ))

    # Проверка описания.
    description = card.get("description")

    if not isinstance(description, str):
        errors.append((
            "description_type",
            "description должен быть строкой",
        ))
    else:
        description = description.strip()

        if not 120 <= len(description) <= 700:
            errors.append((
                "description_length",
                "Длина description должна быть 120–700 символов",
            ))

        sentence_count = count_sentences(description)
        if not 2 <= sentence_count <= 5:
            errors.append((
                "sentence_count",
                "description должен содержать от 2 до 5 предложений",
            ))

    # Проверка массивов: диапазон количества и непустые строки.
    for field, (min_count, max_count) in ARRAY_LENGTH_RANGES.items():
        value = card.get(field)

        if not isinstance(value, list):
            errors.append((
                f"{field}_type",
                f"{field} должен быть массивом",
            ))
            continue

        if not min_count <= len(value) <= max_count:
            errors.append((
                f"{field}_count",
                f"{field} должен содержать от {min_count} до "
                f"{max_count} элементов",
            ))

        for index, item in enumerate(value):
            if not isinstance(item, str):
                errors.append((
                    f"{field}_item_type",
                    f"{field}[{index}] должен быть строкой",
                ))
            elif not item.strip():
                errors.append((
                    f"{field}_empty_item",
                    f"{field}[{index}] не должен быть пустым",
                ))

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

    raise ValueError("raw_output отсутствует или имеет неверный тип")


def main():
    args = parse_args()
    path = Path(args.input)

    if not path.exists():
        print(f"Файл не найден: {args.input}")
        return

    records = []

    # Некорректные строки JSONL сохраняются как отдельные невалидные записи.
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                records.append({
                    "_parse_error": f"Строка {line_number}: {exc}",
                })

    valid = 0
    invalid = 0
    errors_by_reason = Counter()

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
                raise ValueError("Запись JSONL должна быть объектом")

            if "_parse_error" in record:
                raise json.JSONDecodeError(
                    record["_parse_error"], "", 0
                )

            card = load_card(record)
            errors = validate_card(card, product_id)

        except json.JSONDecodeError as exc:
            errors = [("json_parse", f"Ошибка разбора JSON: {exc}")]
        except (ValueError, TypeError) as exc:
            errors = [("record_error", f"Ошибка чтения карточки: {exc}")]

        if errors:
            invalid += 1
            for code, message in errors:
                errors_by_reason[code] += 1
                print(f"  ОШИБКА [{code}]: {message}")
        else:
            valid += 1
            print("  OK")

    total = len(records)
    valid_percent = valid / total * 100 if total else 0

    print("\n" + "=" * 40)
    print("ИТОГИ ПРОВЕРКИ")
    print("=" * 40)
    print(f"Всего карточек: {total}")
    print(f"Валидных: {valid}")
    print(f"Невалидных: {invalid}")
    print(f"Доля валидных: {valid_percent:.1f}%")
    print(f"Всего нарушений: {sum(errors_by_reason.values())}")

    print("\nНарушения по правилам:")
    for reason, count in sorted(errors_by_reason.items()):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
