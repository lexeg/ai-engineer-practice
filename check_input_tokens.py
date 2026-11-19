
import json
from transformers import AutoTokenizer
from run_batch import MODEL, SYSTEM_PROMPT

tokenizer = AutoTokenizer.from_pretrained(MODEL)

with open("datasets/dev.jsonl", encoding="utf-8") as file:
    products = [json.loads(line) for line in file]

results = []

for product in products:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(product, ensure_ascii=False)
        }
    ]

    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )

    results.append((product["product_id"], product["title"], len(tokens)))

for product_id, title, count in results:
    status = "OK" if count <= 1500 else "LIMIT EXCEEDED"
    print(f"{product_id} | {count:5} tokens | {status} | {title}")

print("\nИтоги:")
print(f"Всего товаров: {len(results)}")
print(f"До 1500 токенов: {sum(count <= 1500 for _, _, count in results)}")
print(f"Больше 1500 токенов: {sum(count > 1500 for _, _, count in results)}")
print(f"Максимум: {max(count for _, _, count in results)}")
