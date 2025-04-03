import pathlib
from time import time

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
# Импортируем PyTorch XLA:
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from torch.cuda.amp import GradScaler, autocast

# Ваши импорты для кастомных модулей...
from gm.layers.pseudo_layers.argument_parsing_strategy.argument_parsing_strategy import ArgumentParsingStrategy
from gm.layers.pseudo_layers.pseudo_linear import PseudoLinear
from gm.layers.pseudo_layers.transformers.pseudo_conv1d import PseudoConv1D
from gm.layers.weights_storage.weights_storage import WeightsStorage
from gm.layers.weights_storage.lora_weights_storage import LoRAWeightsStorage
from gm.lora.init_strategy.lora_full_init_strategy import LoRAFullInitStrategy
from gm.lora.lora import LoRA
from gm.pseudo_model import PseudoModule
from gm.settings import CPU_DEVICE, CUDA_DEVICE

# Для TPU используем:
# device = xm.xla_device()
device = None
print("Используем TPU устройство:", device)

model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

# Новый специальный токен – разделитель вопроса и ответа:
SEP_TOKEN = "<|QA_SEP|>"
END_TOKEN = "<|endoftext|>"

# Добавляем спецтокен в токенизатор:
special_tokens_dict = {"additional_special_tokens": [SEP_TOKEN]}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
SEP_TOKEN_ID = tokenizer.convert_tokens_to_ids(SEP_TOKEN)

# --- Функции для подготовки данных (без изменений) ---
def create_tokenized_data(dialog_messages, tokenizer, sep_token, max_length=512):
    train_samples = []
    for i in range(0, len(dialog_messages), 2):
        user_msg = dialog_messages[i]
        if i + 1 < len(dialog_messages):
            assistant_msg = dialog_messages[i + 1]
        else:
            break
        if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
            prompt_text = user_msg["content"]
            answer_text = assistant_msg["content"]
            full_text = prompt_text + " " + sep_token + " " + answer_text
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length
            )
            train_samples.append(tokenized)
    return train_samples

def create_training_dataset(dataset, tokenizer, sep_token="<|QA_SEP|>", use_pretokenized=False, max_length=4096, mask_prompt: bool = True):
    def build_prompt(ex):
        qa_text = ex['conversation']
        return {'text': f'{qa_text[0]["content"]}{SEP_TOKEN}{qa_text[1]["content"]}{END_TOKEN}'}
    def tokenize_and_create_labels(example):
        text = example["text"]
        tokenized = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
        )
        input_ids_batch = tokenized["input_ids"]
        attention_mask_batch = tokenized["attention_mask"]
        labels_batch = []
        for input_ids in input_ids_batch:
            labels = input_ids.copy()
            if mask_prompt:
                if SEP_TOKEN_ID not in input_ids:
                    print(input_ids)
                    raise Exception("No SEP token")
                answer_start_position = input_ids.index(SEP_TOKEN_ID) + 1
                for i in range(answer_start_position):
                    labels[i] = -100
            labels_batch.append(labels)
        tokenized["labels"] = labels_batch
        return tokenized
    ds = dataset.map(lambda ex: build_prompt(ex), remove_columns=dataset.column_names)
    ds = ds.map(tokenize_and_create_labels, remove_columns=['text'], batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

# Пример создания датасетов (оставляем как есть)
test_dialog_data = [
    {"content": "Привет! У меня зреет идея...", "role": "user"},
    {"content": "Привет! Идея создания нового формата...", "role": "assistant"}
]
dataset = load_dataset('Vikhrmodels/GrandMaster-PRO-MAX')
# train_data = create_training_dataset(dataset['train'], tokenizer, SEP_TOKEN, max_length=512, mask_prompt=False)
train_data = create_training_dataset(Dataset.from_dict(dataset['train'][:100]), tokenizer, SEP_TOKEN, max_length=512, mask_prompt=False)
test_data = create_training_dataset(dataset['test'], tokenizer, SEP_TOKEN, max_length=512, mask_prompt=False)
print("Получили", len(train_data), "примеров для обучения")

# Подготовка модели с LoRA и pseudo-модулями (оставляем без изменений)
init_strategy = LoRAFullInitStrategy(LoRA)
weights_storage = LoRAWeightsStorage(ArgumentParsingStrategy({}), device, init_strategy)
pseudo_model = PseudoModule.create_patched_pseudo_model(
    weights_storage=weights_storage,
    module=model,
    mapping={
        nn.Linear: PseudoLinear,
        # Если используется Conv1D, проверьте, что его поддержка нужна
        # Conv1D: PseudoConv1D,
    },
    target_modules=[
        'k_proj',
        'v_proj',
        'q_proj',
    ],
)
weights_storage.build_storage()

# --- Адаптированный цикл обучения для TPU ---
def tpu_train_loop(rank, pseudo_model, train_dataset, batch_size, lr, num_epochs):
    # Каждое ядро TPU получает свое устройство
    device = xm.xla_device()
    pseudo_model.to(device)
    # Определяем collate_fn – если у вас уже есть кастомная функция, можно оставить её
    def collate_fn(examples):
        return {
            "input_ids": torch.stack([torch.tensor(e["input_ids"]) for e in examples]),
            "labels": torch.stack([torch.tensor(e["labels"]) for e in examples])
        }
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # Если у вас есть attention_mask в данных, добавьте и его в collate_fn
    # Настраиваем оптимизатор для параметров, которые требуют градиент
    optimizer = torch.optim.Adam([p for p in pseudo_model._module.parameters() if p.requires_grad], lr=lr)
    pseudo_model.train()
    scaler = GradScaler()  # если хотите использовать автоматическое масштабирование градиентов
    start_step_time = time()
    batches_count = len(train_loader)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            # Отправляем данные на TPU
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            # Используем autocast с bfloat16 (TPU оптимизированы для bfloat16)
            with autocast(device_type='xla', dtype=torch.bfloat16):
                outputs = pseudo_model._module(inputs, labels=labels)
                loss = outputs.loss
            loss.backward()
            # Вместо optimizer.step(), используем xm.optimizer_step
            xm.optimizer_step(optimizer)
            running_loss += loss.item()
            if step % 10 == 0:
                current_time = time()
                print(f"Epoch {epoch + 1}, Step {step} / {batches_count}, Loss: {running_loss / (step + 1):.4f}, "
                      f"Time: {current_time - start_step_time:.2f} sec")
                start_step_time = current_time
            if step % 1000 == 0:
                pseudo_model._weights_storage.update_weights_and_reinit_lora()
        print(f"Epoch {epoch + 1} finished, avg loss: {running_loss / len(train_loader):.4f}")
        pseudo_model.save_model(f'/content/drive/MyDrive/Colab\ Notebooks/bp/models/neo{epoch}.pth')
    return pseudo_model

# Запускаем тренировку на TPU, используя xmp.spawn для распределенного обучения:
def train_on_tpu():
    batch_size = 2
    lr = 1e-4
    num_epochs = 10
    trained_model = xmp.spawn(
        tpu_train_loop,
        args=(pseudo_model, train_data, batch_size, lr, num_epochs),
        nprocs=8,
        join=True
    )
    return trained_model

if __name__ == '__main__':
    trained_pseudo_model = train_on_tpu()
