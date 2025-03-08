from time import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, PreTrainedTokenizerBase
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

from gm.settings import CPU_DEVICE, CUDA_DEVICE

device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)
# device = torch.device(CPU_DEVICE)
print(device)

# Задаем имя модели и загружаем токенизатор и модель
# model_name = "meta-llama/Llama-3.2-1B"
model_name = "openai-community/gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<|reserved_special_token_0|>"]})
# если модель уже инициализирована, нужно расширить эмбеддинги:
model.resize_token_embeddings(len(tokenizer))
split_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_0|>")
SPLIT_TOKEN = ['<|reserved_special_token_0|>', split_token_id]

# Настраиваем LoRA с помощью PEFT
# Важно: параметр target_modules подбирается под архитектуру модели
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # для генеративных моделей
    inference_mode=False,  # если планируете дообучение
    r=8,  # размерность LoRA (rank)
    lora_alpha=32,  # масштабирующий коэффициент
    lora_dropout=0.1,  # dropout в LoRA слоях
    target_modules=["c_attn"]  # для GPT-2 обычно целевыми являются слои внимания ("c_attn")
)

# Применяем LoRA к модели
model = get_peft_model(model, peft_config)
print("Параметры модели с PEFT:")
model.print_trainable_parameters()

# Загружаем датасет (например, wikitext-2)
dataset = load_dataset("openai/gsm8k", "main")


def build_prompt(
        example,
        question_key="question",
        answer_key="answer",
):
    """
    Формирует одну строку из вопроса и ответа.
    Можно расширить системными сообщениями, спецтокенами и т.д.
    """
    q = example[question_key]
    a = example[answer_key]

    # Пример: "User: Как дела?\nAssistant: Всё хорошо..."
    text = f"{q}{SPLIT_TOKEN[0]}{a}"

    return {"text": text}


def tokenize_and_create_labels(
        examples,
        tokenizer: PreTrainedTokenizerBase,
        max_length=128,
        mask_prompt: bool = True
):
    """
    1) Применяет tokenizer к полю 'text'
    2) Создаёт labels = input_ids (в случае casual LM)
    3) (опционально) маскирует вопрос, чтобы не учитывать его в лоссе
    """

    # Шаг токенизации
    tokenized = tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding=True,
    )

    input_ids_batch = tokenized["input_ids"]
    attention_mask_batch = tokenized["attention_mask"]

    # Создаём labels = копия input_ids
    labels_batch = []
    for input_ids in input_ids_batch:
        labels = input_ids.copy()

        if mask_prompt:
            if SPLIT_TOKEN[1] not in input_ids:
                print(input_ids)
                exit(-1)
            answer_start_position = input_ids.index(SPLIT_TOKEN[1]) + 1
            if answer_start_position is None: answer_start_position = 0
            for i in range(answer_start_position):
                labels[i] = -100

        labels_batch.append(labels)

    tokenized["labels"] = labels_batch
    return tokenized


def prepare_dataset(
        dataset,
        tokenizer,
        question_key="question",
        answer_key="answer",
        max_length=128,
        mask_prompt=True
):
    # 1) Склеим поля question/answer в одну строку "text"
    ds = dataset.map(
        lambda ex: build_prompt(ex, question_key=question_key, answer_key=answer_key),
        remove_columns=dataset.column_names
    )

    # 2) Токенизация + генерация labels
    ds = ds.map(
        lambda ex: tokenize_and_create_labels(
            ex,
            tokenizer=tokenizer,
            max_length=max_length,
            mask_prompt=mask_prompt
        ),
        remove_columns=['text'],
        batched=True,
    )
    return ds


train_dataloader = DataLoader(
    dataset,
    shuffle=True,
)
datasets = train_dataloader.dataset
train_ds = prepare_dataset(datasets["train"], tokenizer, max_length=256)
test_ds = prepare_dataset(datasets["test"], tokenizer, max_length=256)

# Задаем параметры обучения
training_args = TrainingArguments(
    output_dir="./peft_finetuned_model",  # директория для сохранения чекпоинтов
    per_device_train_batch_size=2,  # размер батча (подберите под вашу GPU)
    num_train_epochs=1,  # количество эпох (для теста можно поставить 1)
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    evaluation_strategy="no"
)

# Создаем Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
)

# Запускаем обучение
# trainer.train()

# Сохраняем дообученную модель (с параметрами PEFT)
model.save_pretrained("./peft_finetuned_model")
print("Финальная модель сохранена в './peft_finetuned_model'")

model.eval()
start_inp = "What is the capital of America?"
inputs = tokenizer(start_inp, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
start = time()
outputs = model.generate(**inputs, max_length=50, use_cache=False)
print(time() - start)
print(tokenizer.decode(outputs[0]))
