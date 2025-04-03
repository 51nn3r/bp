import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, Conv1D
from datasets import load_dataset, Dataset as D
from torchmetrics import Accuracy

from gm.layers.pseudo_layers.argument_parsing_strategy.argument_parsing_strategy import ArgumentParsingStrategy
from gm.layers.pseudo_layers.pseudo_linear import PseudoLinear
from gm.layers.pseudo_layers.transformers.pseudo_conv1d import PseudoConv1D
from gm.layers.weights_storage.lora_weights_storage import LoRAWeightsStorage
from gm.lora.init_strategy.lora_full_init_strategy import LoRAFullInitStrategy
from gm.lora.lora import LoRA
from gm.pseudo_model import PseudoModule
from gm.utils import create_glue_dataset

from gm.settings import CPU_DEVICE, CUDA_DEVICE

device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)

model_name = 'FacebookAI/xlm-roberta-large'
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, use_fast=True)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

print(model)

tokenizer.pad_token = tokenizer.eos_token
END_TOKEN = "<|endoftext|>"
SEP_TOKEN = "<|QA_SEP|>"
special_tokens_dict = {"additional_special_tokens": [SEP_TOKEN]}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
END_TOKEN_ID = tokenizer.convert_tokens_to_ids(END_TOKEN)
SEP_TOKEN_ID = tokenizer.convert_tokens_to_ids(SEP_TOKEN)

metric = Accuracy(task="multiclass", num_classes=2)

weights_storage = LoRAWeightsStorage(ArgumentParsingStrategy({}), LoRAFullInitStrategy(LoRA), device=device,
                                     dtype=torch.float32)

pseudo_model = PseudoModule.create_patched_pseudo_model(
    weights_storage=weights_storage,
    module=model,
    mapping={
        nn.Linear: PseudoLinear,
        Conv1D: PseudoConv1D,
    },
    target_modules=[
        'query',
        'key',
        'value',
    ],
)

weights_storage.build_storage(4, 32, torch.float32, 0.1)

MAX_LENGTH = 128
TASK = 'sst2'
dataset = load_dataset("nyu-mll/glue", TASK)

train_data = create_glue_dataset(dataset['train'], task=TASK, tokenizer=tokenizer, max_length=MAX_LENGTH)
test_data = create_glue_dataset(dataset['validation'], task=TASK, tokenizer=tokenizer, max_length=MAX_LENGTH)

print(model)

w1 = [p for p in weights_storage._lora_modules[0][0].prarmeters()]
weights_storage.update_weights_and_reinit_lora()
w2 = [p for p in weights_storage._lora_modules[0][0].prarmeters()]
print(w1)
print('-' * 100)
print(w2)

'''
pseudo_model.fit(
    train_dataset=train_data,
    test_dataset=test_data,
    batch_size=4,
    lr=1e-4,
    num_epochs=10,
    device=device,
)
'''
