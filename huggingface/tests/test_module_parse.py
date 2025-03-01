from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# model_name = "meta-llama/Llama-3.2-1B"
model_name = "openai-community/gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model: nn.Module = AutoModelForCausalLM.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
print(config.hidden_size)

from transformers.pytorch_utils import Conv1D

def parse(name, module: nn.Module):
    if not [child for child in module.children()]:
        print(name, module.__class__, isinstance(module, Conv1D), [f'{name}: {param.shape};' for name, param in module.named_parameters()])
    else:
        print(name, module.__class__)

    for name, sub_module in module.named_children():
        parse(name, sub_module)


parse('model', model)

# print(model.state_dict())
