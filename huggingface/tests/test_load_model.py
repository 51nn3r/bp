import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from time import time

from transformers.pytorch_utils import Conv1D

from gm.layers.pseudo_layers.argument_parsing_strategy.argument_parsing_strategy import ArgumentParsingStrategy
from gm.layers.pseudo_layers.pseudo_linear import PseudoLinear
from gm.layers.pseudo_layers.transformers.pseudo_conv1d import PseudoConv1D
from gm.layers.weights_storage.lora_weights_storage import LoRAWeightsStorage
from gm.layers.weights_storage.weights_storage import WeightsStorage
from gm.lora.init_strategy.lora_full_init_strategy import LoRAFullInitStrategy
from gm.lora.lora import LoRA
from gm.pseudo_model import PseudoModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

torch.manual_seed(32)

# model_name = "meta-llama/Llama-3.2-1B"
# model_name = "openai-community/gpt2-medium"
model_name = "EleutherAI/gpt-neo-2.7B"
config = AutoConfig.from_pretrained(model_name)
config._attn_implementation = "flash_attention_2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    config=config
)

''''''
init_strategy = LoRAFullInitStrategy(LoRA)
weights_storage = LoRAWeightsStorage(ArgumentParsingStrategy({}), init_strategy, device)

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
model.to(device)

# start_inp = "The mean and standard-deviation are calculated per-dimension over "
# start_inp = "What is the capital of America?"
start_inp = "Привет, как дела?"

inputs = tokenizer(start_inp, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
print(inputs)
start = time()
'''
outputs = model.generate(**inputs, max_length=50, use_cache=False)
print(time() - start)
print(tokenizer.decode(outputs[0]))
'''

'''
def print_module(module: nn.Module):
    print(module.__class__)
    for child in module.children():
        print_module(child)


print_module(model)
print(model)
'''
