import logging
import os
import math
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM

# CONFIGURATION
MODEL_NAME = "EleutherAI/gpt-neo-125m"
DATASET_NAME = "TUKE-DeutscheTelekom/skquad"
# DATASET_NAME = "rajpurkar/squad_v2"
DIR_PREFIX = "./final_tests"
OUTPUT_DIR = f"{DIR_PREFIX}/full_finetune"
LOG_FILE = f"{OUTPUT_DIR}/training_log.txt"
LAST_EPOCH = 0
CKPT_PATH = f"{OUTPUT_DIR}/epoch{LAST_EPOCH}.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(DEVICE)
BATCH_SIZE = 16  # Reduced due to memory constraints
EPOCHS = 5
LEARNING_RATE = 3e-5
GRAD_ACC_STEPS = 4
MAX_PROMPT = 256
MAX_ANSWER = 64
MAX_LENGTH = MAX_PROMPT + MAX_ANSWER + 16

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

SEP_QC = "<|sep_qc|>"
SEP_CA = "<|sep_ca|>"
END_ANS = "<|endofanswer|>"

# 1. LOAD MODEL & TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.add_special_tokens({"additional_special_tokens": [SEP_QC, SEP_CA, END_ANS]})

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

sep_qc_id, sep_ca_id, eoa_id = tokenizer.convert_tokens_to_ids([SEP_QC, SEP_CA, END_ANS])

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# Enable gradient checkpointing to save memory
# model.gradient_checkpointing_enable()


# 2. LOAD & PREPROCESS DATA
def to_samples(examples):
    out_ids, out_labels = [], []
    for question, context, answers in zip(
            examples["question"], examples["context"], examples["answers"]
    ):
        texts = answers.get("text", [])
        if not texts:
            continue

        for answer in texts:
            q_ids = tokenizer.encode(question, add_special_tokens=False)
            c_ids = tokenizer.encode(context, add_special_tokens=False)
            a_ids = tokenizer.encode(answer, add_special_tokens=False)

            prompt = q_ids + [sep_qc_id] + c_ids + [sep_ca_id]
            seq = prompt + a_ids + [tokenizer.eos_token_id] + [eoa_id]
            if len(seq) > MAX_LENGTH:
                seq = seq[:MAX_LENGTH]

            labels = [-100] * len(prompt) + a_ids + [tokenizer.eos_token_id] + [eoa_id]
            labels = labels[:len(seq)]
            out_ids.append(seq)
            out_labels.append(labels)
    return {"input_ids": out_ids, "labels": out_labels}


raw_ds = load_dataset(DATASET_NAME)
train_proc = raw_ds["train"].map(to_samples, batched=True, remove_columns=raw_ds["train"].column_names)
val_proc = raw_ds["validation"].map(to_samples, batched=True, remove_columns=raw_ds["validation"].column_names)
ds_tok = DatasetDict({"train": train_proc, "validation": val_proc})
ds_tok.set_format(type="torch", columns=["input_ids", "labels"])


# 3. COLLATE & DATALOADERS
def collate_fn(batch):
    pad_id = tokenizer.pad_token_id
    input_ids = [ex["input_ids"] for ex in batch]
    labels = [ex["labels"] for ex in batch]

    padded_inputs = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")["input_ids"]
    padded_labels = tokenizer.pad({"input_ids": labels}, padding=True, return_tensors="pt")["input_ids"]
    padded_labels[padded_labels == pad_id] = -100

    return {
        "input_ids": padded_inputs.to(device),
        "attention_mask": (padded_inputs != pad_id).long().to(device),
        "labels": padded_labels.to(device),
    }


train_loader = DataLoader(ds_tok["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(ds_tok["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# 4. OPTIMIZER & SCALER
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scaler = GradScaler() if DEVICE == "cuda" else None

# 5. TRAINING LOOP
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(train_loader, 1):
        with autocast(enabled=(DEVICE == "cuda"), dtype=torch.float32):
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACC_STEPS

        if DEVICE == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % GRAD_ACC_STEPS == 0:
            if DEVICE == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACC_STEPS

        if step % 100 == 0:
            avg_loss = total_loss / step
            ppl = math.exp(avg_loss)
            msg = f"[Epoch {epoch}] step {step}/{len(train_loader)} loss={avg_loss:.4f} ppl={ppl:.1f}"
            logging.info(msg)
            print(msg)

    # Epoch evaluation
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for batch in eval_loader:
            with autocast(enabled=(DEVICE == "cuda"), dtype=torch.float32):
                outputs = model(**batch)
                eval_loss += outputs.loss.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_eval_loss = eval_loss / len(eval_loader)
    msg = (
        f"[Epoch {epoch} Final] Train loss: {avg_train_loss:.4f} "
        f"Eval loss: {avg_eval_loss:.4f}"
    )
    logging.info(msg)
    print(msg)

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"epoch{epoch}.pt"))

# Final save
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
