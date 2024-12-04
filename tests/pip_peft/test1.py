from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

lora_config = LoraConfig(
    task_type="CAUSAL_LM",  # Тип задачи
    inference_mode=False,
    r=16,  # Ранг
    lora_alpha=32,
    lora_dropout=0.1
)

lora_model = get_peft_model(model, lora_config)

lora_model.train()
