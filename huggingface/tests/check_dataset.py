from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("glue", "qnli", split="train")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

max_length = 0
num_examples = 0

for example in dataset:
    question = example["question"]
    sentence = example["sentence"]
    tokens = tokenizer(question, sentence, truncation=False, padding=False)["input_ids"]

    length = len(tokens)
    if length > max_length:
        max_length = length
    num_examples += 1

print(f"Processed {num_examples} examples.")
print(f"Maximum token length found in the dataset: {max_length}")
