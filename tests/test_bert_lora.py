import logging
import os
import math
import torch
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Conv1D
import evaluate
import numpy as np
from collections import defaultdict

from gm.layers.pseudo_layers.argument_parsing_strategy.argument_parsing_strategy import ArgumentParsingStrategy
from gm.layers.pseudo_layers.pseudo_linear import PseudoLinear
from gm.layers.pseudo_layers.transformers.pseudo_conv1d import PseudoConv1D
from gm.layers.weights_storage.configs.lora_weights_storage_config import LoraWeightsStorageConfig
from gm.layers.weights_storage.lora_weights_storage import LoRAWeightsStorage
from gm.lora.enable_strategy.weighted_subset_enable_strategy import WeightedSubsetEnableStrategy
from gm.lora.init_strategy.lora_full_init_strategy import LoRAFullInitStrategy
from gm.lora.lora_svd import LoRA_SVD
from gm.pseudo_model import PseudoModule

#  CONFIGURATION
MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "TUKE-DeutscheTelekom/skquad"
DIR_PREFIX = "./final_tests"
OUTPUT_DIR = f"{DIR_PREFIX}/bert_qa_lora"
LOG_FILE = f"{OUTPUT_DIR}/training_log.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(DEVICE)

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 3e-5
MAX_LENGTH = 384
DOC_STRIDE = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

#  1. LOAD MODEL & TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
model.to(device)

weights_storage = LoRAWeightsStorage(
    LoraWeightsStorageConfig(
        argument_parsing_strategy=ArgumentParsingStrategy({}),
        lora_init_strategy=LoRAFullInitStrategy(LoRA_SVD),
        lora_enable_strategy_cls=WeightedSubsetEnableStrategy,
        device=device,
        dtype=torch.float32,
        enabled_adapters_proportion=1.0,
        rank=32,
        svd_rank=24,
        alpha=32,
        lora_dropout=0.1,
    )
)

pseudo_model = PseudoModule.create_patched_pseudo_model(
    weights_storage=weights_storage,
    module=model,
    mapping={nn.Linear: PseudoLinear, Conv1D: PseudoConv1D},
    target_modules=['query', 'key', 'value'],
)
weights_storage.build_storage()
model.to(device)

for p in model.parameters():
    p.requires_grad = False

weights_storage.train()
weights_storage.enable_grad()

lora_params = [
    p for layers in weights_storage._lora_modules
    for adapter in layers
    if adapter is not None
    for p in adapter.parameters()
    if p.requires_grad
]
head_params = [p for name, p in model.named_parameters() if 'qa_outputs' in name]

#  2. LOAD & PREPROCESS DATA
raw_ds = load_dataset(DATASET_NAME)


def prepare_features(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offsets = tokenized.pop("offset_mapping")
    tokenized["example_id"] = [examples["id"][i] for i in sample_map]

    starts, ends = [], []
    for i, offset in enumerate(offsets):
        sample_idx = sample_map[i]
        ans = examples["answers"][sample_idx]
        if len(ans["answer_start"]) == 0:
            starts.append(0)
            ends.append(0)
        else:
            start_char = ans["answer_start"][0]
            end_char = start_char + len(ans["text"][0])
            seq_ids = tokenized.sequence_ids(i)

            ctx_start = seq_ids.index(1)
            ctx_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)

            tok_start = ctx_start
            while tok_start <= ctx_end and offset[tok_start][0] <= start_char:
                tok_start += 1
            starts.append(tok_start - 1)

            tok_end = ctx_end
            while tok_end >= ctx_start and offset[tok_end][1] >= end_char:
                tok_end -= 1
            ends.append(tok_end + 1)

    tokenized["offset_mapping"] = offsets
    tokenized["start_positions"] = starts
    tokenized["end_positions"] = ends
    return tokenized


train_ds = raw_ds["train"].map(
    prepare_features,
    batched=True,
    remove_columns=raw_ds["train"].column_names,
)
val_ds = raw_ds["validation"].map(
    prepare_features,
    batched=True,
    remove_columns=raw_ds["validation"].column_names,
)
val_features = raw_ds["validation"].map(
    prepare_features,
    batched=True,
    remove_columns=raw_ds["validation"].column_names,
)

train_ds.set_format(type="torch",
                    columns=["input_ids", "attention_mask", "start_positions", "end_positions", "example_id"])
val_ds.set_format(type="torch",
                  columns=["input_ids", "attention_mask", "start_positions", "end_positions", "example_id"])


#  3. DATALOADERS
def collate_fn(batch):
    input_ids = [ex["input_ids"] for ex in batch]
    masks = [ex["attention_mask"] for ex in batch]
    starts = [ex["start_positions"] for ex in batch]
    ends = [ex["end_positions"] for ex in batch]
    example_ids = [ex["example_id"] for ex in batch]

    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": masks},
        padding=True,
        return_tensors="pt"
    )
    return {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "start_positions": torch.tensor(starts, dtype=torch.long),
        "end_positions": torch.tensor(ends, dtype=torch.long),
        "example_id": example_ids,
    }


train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#  4. OPTIMIZER & SCALER
for p in head_params:
    p.requires_grad = True
    pass

optimizer = AdamW([
    {"params": lora_params, "lr": LEARNING_RATE, "weight_decay": 0.05},
    {"params": head_params, "lr": LEARNING_RATE, "weight_decay": 0.01},
])
scaler = GradScaler() if DEVICE.startswith("cuda") else None

weights_storage.reset_lora()

#  5. TRAINING LOOP
for epoch in range(1, EPOCHS + 1):
    model.train()
    weights_storage.train()

    running_loss = 0.0
    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_pos = batch["start_positions"].to(device)
        end_pos = batch["end_positions"].to(device)

        # forward + backward
        if scaler:
            with torch.cuda.amp.autocast():
                out = model(
                    input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_pos,
                    end_positions=end_pos,
                )
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(
                input_ids,
                attention_mask=attention_mask,
                start_positions=start_pos,
                end_positions=end_pos,
            )
            out.loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        running_loss += out.loss.item()

        if step % 100 == 0:
            avg = running_loss / step
            msg = f"[Epoch {epoch}] step {step}/{len(train_loader)} loss={avg:.4f}"
            logging.info(msg)
            print(msg)

    all_start_logits, all_end_logits, all_feature_example_ids = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            all_start_logits.append(out.start_logits.cpu().numpy())
            all_end_logits.append(out.end_logits.cpu().numpy())
            all_feature_example_ids.extend(batch["example_id"])

    start_logits = np.concatenate(all_start_logits, axis=0)
    end_logits = np.concatenate(all_end_logits, axis=0)

    feat_idxs_per_example = defaultdict(list)
    for feat_idx, ex_id in enumerate(all_feature_example_ids):
        feat_idxs_per_example[ex_id].append(feat_idx)

    dataset_name = DATASET_NAME
    metric = evaluate.load("squad_v2" if "v2" in dataset_name else "squad")

    for ex in raw_ds["validation"]:
        ex_id = ex["id"]
        golds = ex["answers"]["text"]
        if len(golds) == 0:
            continue

        best_score = -1e9
        best_pred = ""

        for feat_idx in feat_idxs_per_example[ex_id]:
            s_logits = start_logits[feat_idx]
            e_logits = end_logits[feat_idx]
            s_pos = int(s_logits.argmax())
            e_pos = int(e_logits.argmax())
            score = float(s_logits[s_pos] + e_logits[e_pos])
            if e_pos < s_pos or score <= best_score:
                continue

            offsets = val_features[feat_idx]["offset_mapping"]
            off_s = offsets[s_pos]
            off_e = offsets[e_pos]
            if off_s is None or off_e is None:
                continue
            start_char, end_char = off_s[0], off_e[1]
            context = ex["context"]
            pred_text = context[start_char:end_char]
            best_score = score
            best_pred = pred_text

        if best_pred is None:
            best_pred = ""

        metric.add(
            prediction={"id": ex_id, "prediction_text": best_pred},
            reference={
                "id": ex_id,
                "answers": {"text": golds, "answer_start": ex["answers"]["answer_start"]},
            },
        )

    res = metric.compute()
    msg = "=== VAL METRICS ===\n" \
          f"Exact Match: {res['exact_match']:.2f}\n" \
          f"F1-score   : {res['f1']:.2f}\n"
    logging.info(msg)
    print(msg)

    pseudo_model.save_model(os.path.join(OUTPUT_DIR, f"epoch{epoch}.pt"))

#  6. FINAL SAVE
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
