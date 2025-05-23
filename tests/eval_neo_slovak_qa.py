# Configuration
from torch import nn

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

BASE_MODEL = "EleutherAI/gpt-neo-2.7B"
CKPT_PATH = "./final_tests/06_xgblora_b6_ft/epoch3.pt"
N_SAMPLES = 200
MAX_NEW_TOKS = 32
SHOW_CASES = True  # print each example
# End configuration

import os, random, torch, evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Conv1D

# Special separators (must match training)
SEP_QC = "<|sep_qc|>"  # between question and context
SEP_CA = "<|sep_ca|>"  # between context and answer
END_ANS = "<|endofanswer|>"

def strip(text: str) -> str:
    """Clean up generated text: remove everything after EOS/newline."""
    return text.split("</s>")[0].split("\n")[0].strip(" .,\n")


# Load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
# add separators if not already present
tok.add_special_tokens({"additional_special_tokens": [SEP_QC, SEP_CA]})
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

sep_qc_id, sep_ca_id, eoa_id = tok.convert_tokens_to_ids([SEP_QC, SEP_CA, END_ANS])
# load base model and then checkpoint
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
)

weights_storage = LoRAWeightsStorage(
    LoraWeightsStorageConfig(
        argument_parsing_strategy=ArgumentParsingStrategy({}),
        lora_init_strategy=LoRAFullInitStrategy(LoRA_SVD),
        lora_enable_strategy_cls=WeightedSubsetEnableStrategy,
        # lora_enable_strategy_cls=LoraEnableAllStrategy,
        device=device,
        dtype=torch.float32,
        enabled_adapters_proportion=1 / 1,
        rank=8,
        svd_rank=6,
        alpha=16,
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

# resize embeddings for new tokens
model.resize_token_embeddings(len(tok))
if CKPT_PATH and os.path.isfile(CKPT_PATH):
    # state = torch.load(CKPT_PATH, map_location="cpu")
    # missing, unexpected = model.load_state_dict(state, strict=False)
    missing, unexpected = pseudo_model.load_model(CKPT_PATH)
    print(f"Loaded checkpoint '{CKPT_PATH}': missing keys {len(missing)}, unexpected {len(unexpected)}")

model.to(device).eval()

# two custom test cases
custom = [
    {
        "id": "custom1",
        "question": "Kde leží hory Tatry?",
        "context": "Vysoké Tatry sú pohorie na severe Slovenska a juhu Poľska.",
        "answer": "na severe Slovenska a juhu Poľska"
    },
    {
        "id": "custom2",
        "question": "Kto napísal knihu Malý princ?",
        "context": "Autorom tejto knihy je francúzsky spisovateľ Antoine de Saint-Exupéry.",
        "answer": "Antoine de Saint-Exupéry"
    }
]
print("\n=== CUSTOM TESTS ===")
for ex in custom:
    q_ids = tok.encode(ex["question"], add_special_tokens=False)
    c_ids = tok.encode(ex["context"], add_special_tokens=False)
    prompt_ids = q_ids + [sep_qc_id] + c_ids + [sep_ca_id]
    inp = torch.tensor([prompt_ids], device=device)
    msk = torch.ones_like(inp)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            out_ids = model.generate(
                inp,
                attention_mask=msk,
                max_new_tokens=MAX_NEW_TOKS + 1,
                do_sample=False,
                eos_token_id=eoa_id,
                pad_token_id=tok.pad_token_id,
            )[0, inp.size(-1):]
    pred = strip(tok.decode(out_ids.tolist()))
    print("Question:", ex["question"])
    print("Prediction:", pred)
    print("Expected :", ex["answer"], "\n")

# prepare evaluation metric and data
ds = load_dataset("TUKE-DeutscheTelekom/squad-sk", split="validation")
metric = evaluate.load("squad")

# sample indices
indices = random.sample(range(len(ds)), N_SAMPLES) if N_SAMPLES else range(len(ds))

# evaluate on validation set
for i in indices:
    ex = ds[i]
    if not ex["answers"]["text"]:
        continue

    qid = ex.get("id", str(i))
    # build token prompt: Q + sep_qc + C + sep_ca
    q_ids = tok.encode(ex["question"], add_special_tokens=False)
    c_ids = tok.encode(ex["context"], add_special_tokens=False)
    prompt_ids = q_ids + [sep_qc_id] + c_ids + [sep_ca_id]
    inp = torch.tensor([prompt_ids], device=device)
    msk = torch.ones_like(inp)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            out_ids = model.generate(
                inp,
                attention_mask=msk,
                max_new_tokens=MAX_NEW_TOKS,
                do_sample=False,
                eos_token_id=eoa_id,
                pad_token_id=tok.pad_token_id,
            )[0, inp.size(-1):]
    pred = strip(tok.decode(out_ids.tolist()))
    gold = ex["answers"]["text"][0]
    metric.add(
        prediction={"id": qid, "prediction_text": pred},
        reference={"id": qid, "answers": {"text": [gold], "answer_start": [0]}},
    )
    if SHOW_CASES:
        print("-" * 60)
        print("Q:", ex["question"])
        print("Pred:", pred)
        print("Gold:", gold)

# compute and print metrics
res = metric.compute()
print("\n=== VAL METRICS ===")
print(f"Exact Match: {res['exact_match']:.2f}")
print(f"F1-score   : {res['f1']:.2f}")
