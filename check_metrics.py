
import json
import re
from pathlib import Path


INPUT_FILE = "outputs/raw_predictions.jsonl"

REQUIRED_FIELDS = {
    "product_id",
    "description",
    "pros",
    "cons",
    "tags"
}


def parse_model_json(raw_output):
    """Извлекает JSON из ответа модели."""

    if not isinstance(raw_output, str):
        raise ValueError("Ответ модели должен быть строкой")

    text = raw_output.strip()

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"\s*```\s*$", "", text)

    return json.loads(text)


def count_sentences(text):
    """Подсчитывает окончания предложений, не считая десятичные точки."""

    if not isinstance(text, str):
        return 0

    return len(re.findall(r"\.(?!\d)|[!?]+", text))


def validate_card(card, expected_product_id):
    """Проверяет карточку по диапазонам требований задания."""

    errors = []

    if not isinstance(card, dict):
        return ["Карточка должна быть JSON-объектом"]

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

    if card.get("product_id") != expected_product_id:
        errors.append("product_id не совпадает с исходным")

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

        if not 2 <= sentence_count <= 5:
            errors.append(
                "description должен содержать 2–5 предложений"
            )

    limits = {
        "pros": (2, 5),
        "cons": (1, 3),
        "tags": (3, 8)
    }

    for field, (minimum, maximum) in limits.items():
        value = card.get(field)

        if not isinstance(value, list):
            errors.append(f"{field} должен быть массивом")
            continue

        if not minimum <= len(value) <= maximum:
            errors.append(
                f"{field} должен содержать от "
                f"{minimum} до {maximum} элементов"
            )

        for index, item in enumerate(value):
            if not isinstance(item, str):
                errors.append(
                    f"{field}[{index}] должен быть строкой"
                )
            elif not item.strip():
                errors.append(
                    f"{field}[{index}] не должен быть пустым"
                )

        valid_items = [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]

        if field == "cons":
            placeholders = {
                "нет",
                "нет недостатков",
                "недостатки отсутствуют",
                "недостатков нет",
                "не обнаружено",
                "не выявлено"
            }

            for item in valid_items:
                if item.strip().lower() in placeholders:
                    errors.append(
                        "cons не должен содержать заглушки "
                        "вместо описания недостатка"
                    )
                    break

        if len(valid_items) != len(set(valid_items)):
            errors.append(f"{field} содержит дубликаты")

    return errors


def load_card(record):
    """Получает исходную карточку без нормализации."""

    raw_output = record.get("raw_output")

    if isinstance(raw_output, dict):
        return raw_output

    if isinstance(raw_output, str):
        return parse_model_json(raw_output)

    raise ValueError("raw_output отсутствует или имеет неверный тип")


def main():
    path = Path(INPUT_FILE)

    if not path.exists():
        print(f"Файл не найден: {INPUT_FILE}")
        return

    with open(path, "r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    valid = 0
    invalid = 0

    print(f"Загружено карточек: {len(records)}")

    for index, record in enumerate(records, start=1):
        product_id = record.get("product_id")

        print(f"\nКарточка №{index} ({product_id})")

        try:
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
