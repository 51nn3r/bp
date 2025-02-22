import torch
from torch.nn.modules import MultiheadAttention
from transformers import AutoTokenizer, AutoModelForCausalLM

from time import time

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
'''
model.to(device)
inputs = tokenizer("playing dnd is", return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
start = time()
outputs = model.generate(**inputs, max_length=50)
print(time() - start)
print(tokenizer.decode(outputs[0]))
'''
sd = model.state_dict()

for name, tensor in sd.items():
    print(name, tensor.shape)

inputs = tokenizer("The Llama 3.2 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction-tuned generative models in 1B and 3B sizes (text in/text out). The Llama 3.2 instruction-tuned text only models are optimized for multilingual dialogue use cases, including agentic retrieval and summarization tasks. They outperform many of the available open source and closed chat models on common industry benchmarks", return_tensors="pt")
print(inputs)
inputs = tokenizer("Llama 3.2 kolekcia viacjazyčných veľkých jazykových modelov (LLM) je kolekcia vopred pripravených a inštrukciami vyladených generatívnych modelov vo veľkostiach 1B a 3B (textový vstup/textový výstup). Modely Llama 3.2, ladené len s textom, sú optimalizované pre prípady použitia viacjazyčného dialógu, vrátane úloh agentného vyhľadávania a sumarizácie. Prekonajú mnohé dostupné modely s otvoreným zdrojom a uzavreté chatovacie modely v bežných priemyselných benchmarkoch", return_tensors="pt")
print(inputs)

