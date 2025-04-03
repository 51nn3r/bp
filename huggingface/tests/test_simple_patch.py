import pathlib
from pickle import FALSE
from time import time

import torch
from bitsandbytes.optim import Adam8bit
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from transformers.pytorch_utils import Conv1D

from gm.lora.compact_lora_linear import CompactLoRALinear, patch_module

from gm.settings import CPU_DEVICE, CUDA_DEVICE

device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)
# device = torch.device(CPU_DEVICE)
print(device)

model_name = "EleutherAI/gpt-neo-2.7B"
# model_name = "EleutherAI/gpt-j-6b"
config = AutoConfig.from_pretrained(model_name)
config._attn_implementation = "flash_attention_2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    config=config
)
if device is not None:
    model.to(device)

tokenizer.pad_token = tokenizer.eos_token
PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

# Новый специальный токен - разделитель вопроса и ответа:
SEP_TOKEN = "<|QA_SEP|>"
END_TOKEN = "<|endoftext|>"

# Сообщаем токенизатору о новом токене:
special_tokens_dict = {"additional_special_tokens": [SEP_TOKEN]}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
SEP_TOKEN_ID = tokenizer.convert_tokens_to_ids(SEP_TOKEN)


def create_tokenized_data(dialog_messages, tokenizer, sep_token, max_length=512):
    """
    Превращает список сообщений вида:
    [
      {"content": "...", "role": "user"},
      {"content": "...", "role": "assistant"},
      ...
    ]
    в список токенизированных примеров:
    "<user_text> <sep_token> <assistant_text>"

    Возвращает список словарей { "input_ids": ..., "attention_mask": ... }.
    """
    train_samples = []

    # Будем идти по сообщениями попарно: user -> assistant
    # Предположим, что диалог разбит корректно (user, assistant, user, assistant, ...)
    # Здесь для простоты - берём подряд user->assistant
    # Если у вас более сложная структура диалога, адаптируйте логику.
    for i in range(0, len(dialog_messages), 2):
        user_msg = dialog_messages[i]

        # Проверяем, чтобы следующий элемент существовал и был "assistant"
        if i + 1 < len(dialog_messages):
            assistant_msg = dialog_messages[i + 1]
        else:
            # Если диалог оканчивается на "user", пропускаем
            break

        # Составляем текст: "вопрос <sep_token> ответ"
        if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
            prompt_text = user_msg["content"]
            answer_text = assistant_msg["content"]

            full_text = prompt_text + " " + sep_token + " " + answer_text

            # Токенизируем
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length
            )

            train_samples.append(tokenized)
        else:
            # Если порядок не соответствует user->assistant, пропустим
            continue

    return train_samples


def create_training_dataset(
        dataset,
        tokenizer,
        sep_token="<|QA_SEP|>",
        use_pretokenized=False,
        max_length=4096,
        mask_prompt: bool = True,
):
    def build_prompt(
            ex,
    ):
        qa_text = ex['conversation']
        return {'text': f'{qa_text[0]["content"]}{SEP_TOKEN}{qa_text[1]["content"]}{END_TOKEN}'}

    def tokenize_and_create_labels(example):
        text = example["text"]
        # Токенизируем
        tokenized = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
        )

        input_ids_batch = tokenized["input_ids"]
        attention_mask_batch = tokenized["attention_mask"]

        # Создаём labels = копия input_ids
        labels_batch = []
        for input_ids in input_ids_batch:
            labels = input_ids.copy()

            if mask_prompt:
                if SEP_TOKEN_ID not in input_ids:
                    print(input_ids)
                    raise "No SEP token"
                answer_start_position = input_ids.index(SEP_TOKEN_ID) + 1
                if answer_start_position is None: answer_start_position = 0
                for i in range(answer_start_position):
                    labels[i] = -100

            labels_batch.append(labels)

        tokenized["labels"] = labels_batch

        return tokenized

    ds = dataset.map(
        lambda ex: build_prompt(ex),
        remove_columns=dataset.column_names
    )

    # Применяем process_example к каждому элементу датасета
    # Возвращаем новый dataset со столбцами "input_ids", "attention_mask"
    ds = ds.map(
        tokenize_and_create_labels,
        remove_columns=['text'],
        batched=True,
    )

    # Устанавливаем формат
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def tokenize_input(inp):
    return tokenizer(
        inp,
        return_tensors="pt",
    )


# prepare model

# Настраиваем LoRA с помощью PEFT
# Важно: параметр target_modules подбирается под архитектуру модели
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # для генеративных моделей
    inference_mode=False,  # если планируете дообучение
    r=8,  # размерность LoRA (rank)
    lora_alpha=32,  # масштабирующий коэффициент
    lora_dropout=0.1,  # dropout в LoRA слоях
    target_modules=[
        'k_proj',
        'v_proj',
        'q_proj',
    ]  # для GPT-2 обычно целевыми являются слои внимания ("c_attn")
)

# Применяем LoRA к модели
model = get_peft_model(model, peft_config)
print("Параметры модели с PEFT:")
model.print_trainable_parameters()
'''
patch_module(
    module=model,
    target_modules=[
        'k_proj',
        'v_proj',
        'q_proj',
    ],
)
'''
'''
'''
test_dialog_data = [
    {
        "content": "Привет! У меня зреет идея...",
        "role": "user"
    },
    {
        "content": "Привет! Идея создания нового формата...",
        "role": "assistant"
    }
]

dataset = load_dataset('Vikhrmodels/GrandMaster-PRO-MAX')
train_data = create_training_dataset(dataset['train'], tokenizer, SEP_TOKEN, max_length=512, mask_prompt=False)
test_data = create_training_dataset(dataset['test'], tokenizer, SEP_TOKEN, max_length=512, mask_prompt=False)
print("Получили", len(train_data), "примеров для обучения")

# pseudo_model.load_model('neo1.pth', device_name=CUDA_DEVICE)
model.gradient_checkpointing_enable()


def fit_model(model: nn.Module, train_dataset, num_epochs=3, batch_size=4, lr=1e-4, device=None):
    if device is None:
        device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)

    # Функция для объединения примеров в батч
    def collate_fn(examples):
        return {
            "input_ids": torch.stack([torch.tensor(e["input_ids"]) for e in examples]),
            "labels": torch.stack([torch.tensor(e["labels"]) for e in examples])
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model.to(device)
    model.train()  # переводим модель в режим обучения

    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            # Можно добавить вывод для отладки:
            # print(f"{name}: requires_grad set to True")
        else:
            param.requires_grad = False
            # Можно добавить вывод для отладки:
            # print(f"{name}: requires_grad set to False")

    param.requires_grad = True

    # Используем Adam8bit (если хотите) – он экономит память на optimizer state.
    optimizer = Adam8bit([p for p in model.parameters() if p.requires_grad], lr=lr)
    # Если предпочитаете стандартный Adam, раскомментируйте:
    # optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    start_step_time = time()
    batches_count = len(train_loader)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            # Если у вас есть attention_mask, можно добавить аналогично:
            if "attention_mask" in batch:
                attention_mask = batch["attention_mask"].to(device)
            else:
                attention_mask = None

            optimizer.zero_grad()

            # Используем autocast для BF16 (если ваша платформа поддерживает BF16)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if attention_mask is not None:
                    outputs = model(inputs, attention_mask=attention_mask, labels=labels)
                else:
                    outputs = model(inputs, labels=labels)
                loss = outputs.loss

            loss.backward()
            total_params_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_params_count += param.grad.numel()

                pass

            print('p> ' + str(total_params_count))
            input('> ')

            optimizer.step()

            running_loss += loss.item()
            if step % 500 == 0:
                current_time = time()
                print(f"Epoch {epoch + 1}, Step {step} / {batches_count}, "
                      f"Loss: {running_loss / (step + 1):.4f}, "
                      f"Time: {current_time - start_step_time:.2f} sec")
                start_step_time = current_time

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} finished, avg loss: {avg_loss:.4f}")


fit_model(
    model,
    train_data,
    batch_size=1,
    lr=1e-4,
    num_epochs=10,
    device=device,
)
'''
text_en = 'What do you think about mushrooms?'
text_ru = 'Что ты думаешь о грибах?'

pseudo_model.eval()
print('[*] processing english text...')
start = time()
inp_en = tokenize_input(text_en)
inp_en = {k: v.to(device) for k, v in inp_en.items()}
out_en = model.generate(**inp_en, max_length=200, use_cache=False)
print(f'{tokenizer.decode(out_en[0])}\ntime: {time() - start}')
print(f'time: {time() - start}')
print('-' * 100)
print('[*] processing russian text...')
start = time()
inp_ru = tokenize_input(text_ru)
inp_ru = {k: v.to(device) for k, v in inp_ru.items()}
out_ru = model.generate(**inp_ru, max_length=200, use_cache=False)
print(f'{tokenizer.decode(out_ru[0])}\ntime: {time() - start}')

print('=' * 100)

pseudo_model.load_model('models/neo_tmp.pth', device_name=CUDA_DEVICE)
pseudo_model.eval()
print('[*] processing english text...')
start = time()
inp_en = tokenize_input(text_en)
inp_en = {k: v.to(device) for k, v in inp_en.items()}
out_en = model.generate(**inp_en, max_length=200, use_cache=False)
print(f'{tokenizer.decode(out_en[0])}\ntime: {time() - start}')
print(f'time: {time() - start}')
print('-' * 100)
print('[*] processing russian text...')
start = time()
inp_ru = tokenize_input(text_ru)
inp_ru = {k: v.to(device) for k, v in inp_ru.items()}
out_ru = model.generate(**inp_ru, max_length=200, use_cache=False)
print(f'{tokenizer.decode(out_ru[0])}\ntime: {time() - start}')
'''
