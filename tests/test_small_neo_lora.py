import logging
import os
import math
import torch
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Conv1D
from bitsandbytes.optim import Adam8bit

from gm.layers.pseudo_layers.argument_parsing_strategy.argument_parsing_strategy import ArgumentParsingStrategy
from gm.layers.pseudo_layers.pseudo_linear import PseudoLinear
from gm.layers.pseudo_layers.transformers.pseudo_conv1d import PseudoConv1D
from gm.layers.weights_storage.configs.lora_weights_storage_config import LoraWeightsStorageConfig
from gm.layers.weights_storage.lora_weights_storage import LoRAWeightsStorage
from gm.lora.enable_strategy.weighted_subset_enable_strategy import WeightedSubsetEnableStrategy
from gm.lora.init_strategy.lora_full_init_strategy import LoRAFullInitStrategy
from gm.lora.lora import LoRA
from gm.lora.lora_svd import LoRA_SVD
from gm.pseudo_model import PseudoModule
from gm.utils.metrics import masked_nll

#  CONFIGURATION
# MODEL_NAME = "EleutherAI/gpt-neo-2.7B"
MODEL_NAME = "EleutherAI/gpt-neo-125m"
# DATASET_NAME = "TUKE-DeutscheTelekom/skquad"
DATASET_NAME = "rajpurkar/squad_v2"
DIR_PREFIX = "./final_tests"
OUTPUT_DIR = f"{DIR_PREFIX}/05_lora_b6_ft"
LOG_FILE = f"{OUTPUT_DIR}/training_log.txt"
LAST_EPOCH = 0
CKPT_PATH = f"{OUTPUT_DIR}/epoch{LAST_EPOCH}.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(DEVICE)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
GRAD_ACC_STEPS = 4  # gradient accumulation steps
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

#  1. LOAD MODEL & TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.add_special_tokens({"additional_special_tokens": [SEP_QC, SEP_CA, END_ANS]})

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

sep_qc_id, sep_ca_id, eoa_id = tokenizer.convert_tokens_to_ids([SEP_QC, SEP_CA, END_ANS])

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
)
model.resize_token_embeddings(len(tokenizer))
'''
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
'''

weights_storage = LoRAWeightsStorage(
    LoraWeightsStorageConfig(
        argument_parsing_strategy=ArgumentParsingStrategy({}),
        lora_init_strategy=LoRAFullInitStrategy(LoRA),
        lora_enable_strategy_cls=WeightedSubsetEnableStrategy,
        # lora_enable_strategy_cls=LoraEnableAllStrategy,
        device=device,
        dtype=torch.float32,
        enabled_adapters_proportion=1 / 1,
        rank=32,
        svd_rank=6,
        alpha=32,
        lora_dropout=0.1,
    )
)

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

if LAST_EPOCH > 0 and os.path.isfile(CKPT_PATH):
    missing, unexpected = pseudo_model.load_model(CKPT_PATH)
    print(f"Loaded checkpoint '{CKPT_PATH}': missing keys {len(missing)}, unexpected {len(unexpected)}")

model.to(device)

for param in model.parameters():
    param.requires_grad = False
    pass

model.train()
weights_storage.train()
weights_storage.enable_grad()

lora_params = [
    param
    for layer_adapters in weights_storage._lora_modules
    for adapter in layer_adapters
    if adapter is not None
    for param in adapter.parameters()
    if param.requires_grad
]

#  2. LOAD & PREPROCESS DATA
raw_ds = load_dataset(DATASET_NAME)  # splits: train, validation, test

unk_id = tokenizer.unk_token_id
if unk_id is None and False:
    print("No unk_token_id")
else:
    def count_unks_in_split(split_name: str):
        total_unk = 0
        total_tokens = 0
        for ex in raw_ds[split_name]:
            for field in ("question", "context"):
                ids = tokenizer.encode(ex[field], add_special_tokens=False)
                total_unk += ids.count(unk_id)
                total_tokens += len(ids)
            for ans in ex["answers"].get("text", []):
                ids = tokenizer.encode(ans, add_special_tokens=False)
                total_unk += ids.count(unk_id)
                total_tokens += len(ids)
        print(
            f"[{split_name:>10}] unk-tokens: {total_unk} "
            f"of {total_tokens} (part {total_unk/total_tokens:.2%})"
        )

    for split in raw_ds.keys():
        count_unks_in_split(split)

def to_samples(examples):
    """
    Explode each example in the batch into one training sample per answer.
    Returns dict of lists for 'input_ids' and 'labels'.
    """
    out_ids, out_labels = [], []

    for question, context, answers in zip(
            examples["question"], examples["context"], examples["answers"]
    ):
        texts = answers.get("text", [])
        if not texts:
            continue  # skip if no answers

        for answer in texts:
            # build prompt
            q_ids = tokenizer.encode(question, add_special_tokens=False)
            c_ids = tokenizer.encode(context, add_special_tokens=False)
            a_ids = tokenizer.encode(answer, add_special_tokens=False)

            prompt = q_ids + [sep_qc_id] + c_ids + [sep_ca_id]
            seq = prompt + a_ids + [tokenizer.eos_token_id] + [eoa_id]
            if len(seq) > MAX_LENGTH:
                seq = seq[-MAX_LENGTH:]

            labels = [-100] * len(prompt) + a_ids + [tokenizer.eos_token_id] + [eoa_id]
            labels = labels[-len(seq):]

            out_ids.append(seq)
            out_labels.append(labels)

    return {"input_ids": out_ids, "labels": out_labels}


# map with batched=True to expand examples
train_proc = raw_ds["train"].map(
    to_samples,
    batched=True,
    remove_columns=raw_ds["train"].column_names,
)
val_proc = raw_ds["validation"].map(
    to_samples,
    batched=True,
    remove_columns=raw_ds["validation"].column_names,
)

ds_tok = DatasetDict({"train": train_proc, "validation": val_proc})
ds_tok.set_format(type="torch", columns=["input_ids", "labels"])

#  3. COLLATE & DATALOADERS
pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id


def collate_fn(batch):
    """
    Pad input_ids and labels to the same length.
    Set label pads to -100 so loss ignores them.
    Build 2D attention mask for non-pad tokens.
    """
    input_ids = [ex["input_ids"] for ex in batch]
    labels = [ex["labels"] for ex in batch]

    padded_inputs = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt"
    )["input_ids"]

    padded_labels = tokenizer.pad(
        {"input_ids": labels},
        padding=True,
        return_tensors="pt"
    )["input_ids"]
    padded_labels[padded_labels == pad_id] = -100

    attention_mask = (padded_inputs != pad_id).long()
    return {
        "input_ids": padded_inputs,
        "attention_mask": attention_mask,
        "labels": padded_labels,
    }


train_loader = DataLoader(
    ds_tok["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)
eval_loader = DataLoader(
    ds_tok["validation"],
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

#  4. OPTIMIZER & SCALER
optimizer = Adam8bit([
    {"params": lora_params, "lr": LEARNING_RATE, "weight_decay": 0.01}
])
use_cuda = DEVICE.startswith("cuda")
scaler = GradScaler() if use_cuda else None

weights_storage.reset_lora()
steps_per_epoch = len(train_loader)
k = 8 * 1 * 2
k = 1
update_freq = EPOCHS * steps_per_epoch // k
# update_freq = 10
print(f'[+] udpate frequency is {update_freq}')

#  5. TRAINING LOOP
for epoch in range(1, EPOCHS + 1):
    model.train()
    weights_storage.train()
    running_nll = 0.0
    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # forward + backward
        if use_cuda:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
                outputs = model(
                    input_ids,
                    attention_mask=mask,
                    labels=labels,
                    use_cache=False,
                )
                # loss = outputs.loss / GRAD_ACC_STEPS
                loss = outputs.loss

            scaler.scale(loss).backward()
        else:
            outputs = model(
                input_ids,
                attention_mask=attention_mask.unsqueeze(1).unsqueeze(2),
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss / GRAD_ACC_STEPS
            loss.backward()

        running_nll += outputs.loss.item()

        # optimizer step
        if step % GRAD_ACC_STEPS == 0:
            if use_cuda:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # logging
        if step % 100 == 0:
            avg_nll = running_nll / step
            ppl = math.exp(avg_nll)
            msg = f"[Epoch {epoch}] step {step}/{len(train_loader)} " \
                  f"avg NLL={avg_nll:.4f} PPL={ppl:.1f}"
            logging.info(msg)
            print(msg)

    # epoch summary
    avg_nll = running_nll / len(train_loader)
    ppl = math.exp(avg_nll)
    msg = f"[Epoch {epoch} END] train NLL={avg_nll:.4f} PPL={ppl:.1f}"
    logging.info(msg)
    print(msg)

    # evaluation
    model.eval()
    eval_nll = 0.0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            if use_cuda:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    out = model(
                        input_ids,
                        attention_mask=attention_mask.unsqueeze(1).unsqueeze(2),
                        labels=labels,
                        use_cache=False,
                    )
                eval_nll += out.loss.item()
            else:
                out = model(
                    input_ids,
                    attention_mask=attention_mask.unsqueeze(1).unsqueeze(2),
                    labels=labels,
                    use_cache=False,
                )
                eval_nll += out.loss.item()

    eval_nll /= len(eval_loader)
    eval_ppl = math.exp(eval_nll)
    msg = f"[Epoch {epoch} EVAL] NLL={eval_nll:.4f} PPL={eval_ppl:.1f}"
    logging.info(msg)
    print(msg)

    # save checkpoint
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"epoch{epoch + LAST_EPOCH}.pt"))
    pseudo_model.save_model()

#  6. FINAL SAVE
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
