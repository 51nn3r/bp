import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time

from transformers.pytorch_utils import Conv1D

from gm.layers.pseudo_layers.argument_parsing_strategy.argument_parsing_strategy import ArgumentParsingStrategy
from gm.layers.pseudo_layers.pseudo_linear import PseudoLinear
from gm.layers.pseudo_layers.transformers.pseudo_conv1d import PseudoConv1D
from gm.layers.weights_storage.weights_storage import WeightsStorage
from gm.pseudo_model import PseudoModule

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

torch.manual_seed(32)

# model_name = "meta-llama/Llama-3.2-1B"
model_name = "openai-community/gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

''''''
weights_storage = WeightsStorage(ArgumentParsingStrategy({}))

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

# start_inp = "The mean and standard-deviation are calculated per-dimension over "
start_inp = "What is the capital of America?"
'''
inputs = tokenizer(start_inp, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
start = time()
outputs = model.generate(**inputs, max_length=50, use_cache=False)
print(time() - start)
print(tokenizer.decode(outputs[0]))
print(model.state_dict)
'''