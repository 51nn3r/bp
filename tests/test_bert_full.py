import logging
import os
import math
import torch
import numpy as np

from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from collections import defaultdict

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)
import evaluate

#  CONFIG 
MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "TUKE-DeutscheTelekom/skquad"
OUTPUT_DIR = "./final_tests/bert_qa_fullfinetune"
LOG_FILE = f"{OUTPUT_DIR}/train.log"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 5
LR = 3e-5
MAX_LENGTH = 384
DOC_STRIDE = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)

#  1. MODEL & TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)

#  2. DATA PREP
raw_ds = load_dataset(DATASET_NAME)


def prepare_features(examples):
    tok = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = tok.pop("overflow_to_sample_mapping")
    offsets = tok.pop("offset_mapping")
    tok["example_id"] = [examples["id"][i] for i in sample_map]

    starts, ends = [], []
    for i, offset in enumerate(offsets):
        ex_idx = sample_map[i]
        ans = examples["answers"][ex_idx]
        if len(ans["answer_start"]) == 0:
            starts.append(0);
            ends.append(0)
        else:
            start_char = ans["answer_start"][0]
            end_char = start_char + len(ans["text"][0])
            seq_ids = tok.sequence_ids(i)
            ctx_start = seq_ids.index(1)
            ctx_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)

            # find token indices
            ts = ctx_start
            while ts <= ctx_end and offset[ts][0] <= start_char:
                ts += 1
            starts.append(ts - 1)

            te = ctx_end
            while te >= ctx_start and offset[te][1] >= end_char:
                te -= 1
            ends.append(te + 1)

    tok["offset_mapping"] = offsets
    tok["start_positions"] = starts
    tok["end_positions"] = ends
    return tok


# map & keep val_features for post‐processing
train_ds = raw_ds["train"].map(prepare_features, batched=True, remove_columns=raw_ds["train"].column_names)
val_features = raw_ds["validation"].map(prepare_features, batched=True,
                                        remove_columns=raw_ds["validation"].column_names)

train_ds = train_ds.remove_columns("offset_mapping")
train_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "start_positions", "end_positions", "example_id"],
)

val_ds = val_features.remove_columns("offset_mapping")
val_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "start_positions", "end_positions", "example_id"],
)


#  3. DATALOADERS
def collate_fn(batch):
    ids = [ex["input_ids"] for ex in batch]
    masks = [ex["attention_mask"] for ex in batch]
    sps = [ex["start_positions"] for ex in batch]
    eps = [ex["end_positions"] for ex in batch]
    eids = [ex["example_id"] for ex in batch]

    padded = tokenizer.pad(
        {"input_ids": ids, "attention_mask": masks},
        padding=True, return_tensors="pt"
    )
    return {
        "input_ids": padded["input_ids"].to(DEVICE),
        "attention_mask": padded["attention_mask"].to(DEVICE),
        "start_positions": torch.tensor(sps, dtype=torch.long, device=DEVICE),
        "end_positions": torch.tensor(eps, dtype=torch.long, device=DEVICE),
        "example_id": eids,
    }


train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#  4. OPTIMIZER & SCALER
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scaler = GradScaler() if DEVICE.startswith("cuda") else None

#  5. TRAIN & EVAL LOOP
metric = evaluate.load("squad_v2" if "v2" in DATASET_NAME else "squad")

for epoch in range(1, EPOCHS + 1):
    # — TRAIN
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(train_loader, start=1):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            start_positions=batch["start_positions"],
            end_positions=batch["end_positions"],
        )
        loss = outputs.loss
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        if step % 100 == 0:
            avg = total_loss / step
            msg = f"[Epoch {epoch}] Step {step}/{len(train_loader)} Loss={avg:.4f}"
            logging.info(msg)
            print(msg)

    # — EVAL: collect logits + example_id
    model.eval()
    all_s, all_e, feat2ex = [], [], []
    with torch.no_grad():
        for batch in eval_loader:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            all_s.append(out.start_logits.cpu().numpy())
            all_e.append(out.end_logits.cpu().numpy())
            feat2ex.extend(batch["example_id"])

    start_logits = np.concatenate(all_s, axis=0)
    end_logits = np.concatenate(all_e, axis=0)

    # group features per example
    idxs_per_ex = defaultdict(list)
    for idx, ex_id in enumerate(feat2ex):
        idxs_per_ex[ex_id].append(idx)

    # compute EM/F1 by simple argmax+offsets
    metric = evaluate.load("squad_v2" if "v2" in DATASET_NAME else "squad")
    for ex in raw_ds["validation"]:
        ex_id = ex["id"];
        golds = ex["answers"]["text"]
        if not golds: continue

        best_score, best_pred = -1e9, ""
        for feat_idx in idxs_per_ex[ex_id]:
            s_log = start_logits[feat_idx];
            e_log = end_logits[feat_idx]
            s_pos = int(s_log.argmax());
            e_pos = int(e_log.argmax())
            score = float(s_log[s_pos] + e_log[e_pos])
            if e_pos < s_pos or score <= best_score: continue

            offs = val_features[feat_idx]["offset_mapping"]
            if offs[s_pos] is None or offs[e_pos] is None: continue
            sc, ec = offs[s_pos][0], offs[e_pos][1]
            pred = ex["context"][sc:ec]
            best_score, best_pred = score, pred

        metric.add(
            prediction={"id": ex_id, "prediction_text": best_pred},
            reference={"id": ex_id, "answers": ex["answers"]},
        )

    res = metric.compute()
    print(f"\n=== Epoch {epoch} VAL === EM={res['exact_match']:.2f}%  F1={res['f1']:.2f}%\n")
    logging.info(f"Epoch {epoch} VAL — EM={res['exact_match']:.2f}, F1={res['f1']:.2f}")

    # save
    model.save_pretrained(f"{OUTPUT_DIR}/epoch{epoch}")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/epoch{epoch}")

#  6. FINAL SAVE
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
