import pathlib
from pickle import FALSE
from time import time

import torch
from datasets import load_dataset, Dataset
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from transformers.pytorch_utils import Conv1D

from gm.layers.pseudo_layers.argument_parsing_strategy.argument_parsing_strategy import ArgumentParsingStrategy
from gm.layers.pseudo_layers.pseudo_linear import PseudoLinear
from gm.layers.pseudo_layers.transformers.pseudo_conv1d import PseudoConv1D
from gm.layers.weights_storage.weights_storage import WeightsStorage
from gm.layers.weights_storage.lora_weights_storage import LoRAWeightsStorage
from gm.lora.init_strategy.lora_full_init_strategy import LoRAFullInitStrategy
from gm.lora.lora import LoRA
from gm.pseudo_model import PseudoModule

from gm.settings import CPU_DEVICE, CUDA_DEVICE

device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)
# device = torch.device(CPU_DEVICE)

model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

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

# prepare model
init_strategy = LoRAFullInitStrategy(LoRA)
weights_storage = LoRAWeightsStorage(ArgumentParsingStrategy({}), device, init_strategy)

pseudo_model = PseudoModule.create_patched_pseudo_model(
    weights_storage=weights_storage,
    module=model,
    mapping={
        nn.Linear: PseudoLinear,
        Conv1D: PseudoConv1D,
    },
    target_modules=[
        'k_proj',
        'v_proj',
        'q_proj',
    ],
)
weights_storage.build_storage()
pseudo_model.fit(
    train_dataset=train_data,
    batch_size=2,
    lr=1e-4,
    num_epochs=10,
    device=device,
)
