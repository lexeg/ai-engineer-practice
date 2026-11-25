# LLM Project

Проект для экспериментов с локальными и облачными языковыми моделями (LLM).

Основные инструменты:
- Python 3.10
- vLLM — запуск локальных моделей и предоставление OpenAI-compatible API
- Hugging Face Transformers
- BitsAndBytes — квантование моделей
- OpenRouter — доступ к облачным LLM
- OpenAI Python SDK — взаимодействие с моделями через API

## 1. Подготовка окружения

### Создание виртуального окружения

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Обновление pip:

```bash
python3 -m pip install pip==24.0
```

### Установка зависимостей

```bash
pip install \
  vllm==0.6.6 \
  transformers==4.47.1 \
  bitsandbytes==0.45.0 \
  accelerate==1.2.1 \
  numpy==1.26.4 \
  jupyter==1.0.0 \
  ipykernel==6.29.4 \
  tqdm==4.66.4
```

Дополнительные зависимости:

```bash
pip install openai
pip install "jinja2>=3.1.6"
```

> Важно: для работы vLLM необходима совместимая конфигурация NVIDIA GPU, драйвера, CUDA и PyTorch. Перед установкой следует проверить совместимость версий.

## 2. Запуск локальной модели через vLLM

Используемая модель:

`Qwen/Qwen2.5-1.5B-Instruct`

### Запуск сервера

Активировать виртуальное окружение:

```bash
source .venv/bin/activate
```

Запустить vLLM в фоновом режиме:

```bash
nohup vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --port 8000 \
  > vllm.log 2>&1 &
```

### Просмотр логов

```bash
tail -f vllm.log
```

### Проверка доступности API

```bash
curl -s http://localhost:8000/v1/models
```

Локальный API доступен по адресу:

`http://localhost:8000/v1`

## 3. Работа с OpenRouter

OpenRouter позволяет обращаться к облачным языковым моделям через единый API.

### Настройка переменных окружения

Указать API-ключ:

```bash
export OPENROUTER_API_KEY="YOUR_API_KEY"
```

Выбрать модель:

```bash
export OPENROUTER_MODEL="particle-ai/particle-1:free"
```

Проверить выбранную модель:

```bash
echo $OPENROUTER_MODEL
```

### Использовавшиеся модели

В ходе экспериментов проверялись следующие модели:

| Модель | Идентификатор OpenRouter |
|---|---|
| GPT OSS 20B | `openai/gpt-oss-20b:free` |
| ERNIE 5.5 Flash | `baidu/ernie-5.5-flash-preview:free` |
| Gemma 4 26B | `google/gemma-4-26b-a4b-it:free` |
| Nemotron 3.5 Lightning | `nvidia/nemotron-3.5-lightning:free` |
| Qwen 2.5 72B | `qwen/qwen2.5-72b-instruct:free` |
| Dots 3 Note | `dots-studio/dots-3-note-preview:free` |
| Particle 1 | `particle-ai/particle-1:free` |

> Доступность моделей и их идентификаторы могут меняться. Актуальную информацию необходимо проверять в каталоге OpenRouter.

## 4. Передача файлов между локальной машиной и сервером

Для передачи файлов используется SCP.

### Локальная машина → сервер

Примеры команд для Windows PowerShell:

```powershell
scp -i .\ai-vm .\files\check_metrics.py ubuntu@SERVER_IP:~/llm-project/

scp -i .\ai-vm .\files\run_batch.py ubuntu@SERVER_IP:~/llm-project/

scp -i .\ai-vm .\files\check_benchmark.py ubuntu@SERVER_IP:~/llm-project/

scp -i .\ai-vm .\files\compare_api.py ubuntu@SERVER_IP:~/llm-project/
```

Передача нескольких файлов выполняется аналогично.

### Сервер → локальная машина

```powershell
scp -i .\ai-vm ubuntu@SERVER_IP:~/llm-project/run_batch.py ./files/run_batch.py

scp -i .\ai-vm ubuntu@SERVER_IP:~/llm-project/check_metrics.py ./files/check_metrics.py

scp -i .\ai-vm ubuntu@SERVER_IP:~/llm-project/outputs/raw_predictions.jsonl ./files/raw_predictions.jsonl
```

### Передача архивов

```powershell
scp -i .\ai-vm ubuntu@SERVER_IP:~/llm-project/llm-project-concurrent.tar .
```

Для других серверов и пользователей необходимо изменить SSH-пользователя, адрес и путь к файлу.

## 5. Архивирование проекта

### Архивирование отдельных компонентов

```bash
tar -cvf llm-project-concurrent.tar \
  check_metrics_80_percent.py \
  outputs/ \
  run_openrouter.py \
  check_benchmark.py \
  datasets/ \
  run_batch.py \
  test_model.py \
  check_input_tokens.py \
  run_batch_80_percent.py \
  test_preprocessing.py \
  check_metrics.py \
  models/ \
  run_batch_benchmark.py
```

### Архивирование всего проекта

Создание архива без виртуального окружения и временных файлов:

```bash
tar -cvf llm-project-concurrent-qwen.tar \
    --exclude='__pycache__' \
    --exclude='.venv' \
    --exclude='llm-project-concurrent-qwen.tar' \
    .
```

Для OpenRouter:

```bash
tar -cvf llm-project-concurrent-openrouter.tar \
    --exclude='__pycache__' \
    --exclude='.venv' \
    --exclude='llm-project-concurrent-openrouter.tar' \
    .
```

### Распаковка архива

```bash
tar -xvf llm-project-concurrent.tar
```

## 6. Полезные команды

### Активация виртуального окружения

```bash
source .venv/bin/activate
```

### Проверка установленного Python

```bash
python --version
```

### Проверка установленных пакетов

```bash
pip list
```

### Проверка GPU

```bash
nvidia-smi
```

### Проверка запущенного vLLM

```bash
curl -s http://localhost:8000/v1/models
```

### Просмотр логов vLLM

```bash
tail -f vllm.log
```

---

## 7. Структура проекта

```text
llm-project/
├── datasets/                 # Наборы данных
├── models/                   # Файлы моделей и вспомогательные данные
├── outputs/                  # Результаты генерации
├── files/                    # Вспомогательные файлы
├── run_batch.py              # Пакетная обработка
├── run_openrouter.py         # Запуск через OpenRouter
├── check_metrics.py          # Проверка метрик
├── check_benchmark.py        # Проверка производительности
├── compare_api.py            # Сравнение API
├── test_model.py             # Проверка модели
└── test_preprocessing.py     # Проверка предобработки
```

*Структура приведена как ориентир: её следует синхронизировать с фактическим содержимым репозитория.*