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
print(device)

# torch.manual_seed(32)

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
weights_storage = LoRAWeightsStorage(ArgumentParsingStrategy({}), device, init_strategy)

pseudo_model = PseudoModule.create_patched_pseudo_model(
    weights_storage=weights_storage,
    module=model,
    mapping={
        nn.Linear: PseudoLinear,
        Conv1D: PseudoConv1D,
    },
)
weights_storage.build_storage(10)

model.to(device)

prev_params = [param for layer in pseudo_model._weights_storage._storage for param in layer]
pseudo_model._weights_storage.update_weights_and_reinit_lora()
params = [param for layer in pseudo_model._weights_storage._storage for param in layer]

for prev, current in zip(prev_params, params):
    print(torch.equal(prev, current))
