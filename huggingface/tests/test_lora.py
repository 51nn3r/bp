import pathlib
from time import time

import torch
from datasets import load_dataset
from torch import nn
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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# model_name = "meta-llama/Llama-3.2-1B"
model_name = "openai-community/gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["[PAD]", "<|reserved_special_token_0|>"]})
# если модель уже инициализирована, нужно расширить эмбеддинги:
model.resize_token_embeddings(len(tokenizer))
split_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_0|>")

''''''
init_strategy = LoRAFullInitStrategy(LoRA)
weights_storage = LoRAWeightsStorage(ArgumentParsingStrategy({}), init_strategy)

pseudo_model = PseudoModule.create_patched_pseudo_model(
    weights_storage=weights_storage,
    module=model,
    mapping={
        nn.Linear: PseudoLinear,
        Conv1D: PseudoConv1D,
    },
)
weights_storage.build_storage()

model.to(device)

path = pathlib.Path().resolve().parent.parent.joinpath('gm/datasets')
# dataset = load_dataset(str(path))
dataset = load_dataset("openai/gsm8k", "main")
# dataset = load_dataset("d0rj/alpaca-cleaned-ru")


SPLIT_TOKEN = ['<|reserved_special_token_0|>', split_token_id]


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

print(train_ds.data)

'''
import code

variables = globals().copy()
variables.update(locals())
shell = code.InteractiveConsole(variables)
shell.interact()
'''
''''''
pseudo_model.fit(
    train_dataset=train_ds,
    vocab_size=128008,
    batch_size=4,
    lr=1e-4,
    num_epochs=3,
)
