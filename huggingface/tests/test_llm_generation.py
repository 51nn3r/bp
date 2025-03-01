import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils_base import BatchEncoding

from time import time

from gm.layers.pseudo_layers.argument_parsing_strategy.argument_parsing_strategy import ArgumentParsingStrategy
from gm.layers.pseudo_layers.pseudo_linear import PseudoLinear
from gm.layers.weights_storage.weights_storage import WeightsStorage
from gm.pseudo_model import PseudoModule


def naive_greedy_decode(
        model: nn.Module,
        batch_encoding: BatchEncoding,
        max_new_tokens: int = 20,
        eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    A simple greedy decoding function that supports a BatchEncoding input.

    Args:
        model (nn.Module): A causal language model (e.g. an instance of AutoModelForCausalLM).
        batch_encoding (BatchEncoding): A BatchEncoding object containing at least "input_ids" and optionally "attention_mask".
        max_new_tokens (int): Maximum number of tokens to generate.
        eos_token_id (int, optional): If provided, generation stops when all examples produce this token.

    Returns:
        torch.Tensor: The generated sequence of input IDs (shape: [batch_size, original_seq_len + generated_tokens]).
    """
    model.eval()
    # Extract input_ids and attention_mask from the BatchEncoding.
    input_ids = batch_encoding["input_ids"]  # shape: [batch_size, seq_len]
    attention_mask = batch_encoding.get("attention_mask", None)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Pass the current sequence to the model.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Get logits for the last token in the sequence.
            next_token_logits = outputs.logits[:, -1, :]  # shape: [batch_size, vocab_size]
            # Greedy selection: choose the token with the highest logit.
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # shape: [batch_size, 1]
            # Append the predicted token to input_ids.
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # If attention_mask is provided, update it by appending ones for new tokens.
            if attention_mask is not None:
                next_mask = torch.ones_like(next_token)
                attention_mask = torch.cat([attention_mask, next_mask], dim=-1)

            # If eos_token_id is specified and all examples produced it, stop generation.
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

    return input_ids


start_inp = "playing dnd is "

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# torch.manual_seed(32)

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

inputs = tokenizer(start_inp, return_tensors="pt")
start = time()
outputs = naive_greedy_decode(model, inputs, 50)
print(time() - start)
print(tokenizer.decode(outputs[0]))

''''''
weights_storage = WeightsStorage(ArgumentParsingStrategy({}))

pseudo_model = PseudoModule.create_patched_pseudo_model(
    weights_storage=weights_storage,
    module=model,
    mapping={nn.Linear: PseudoLinear},
)
weights_storage.build_storage()

model.to(device)

''''''

inputs = tokenizer(start_inp, return_tensors="pt")
start = time()
patched_outputs = naive_greedy_decode(model, inputs, 50)
print(time() - start)
print(tokenizer.decode(patched_outputs[0]))

print(patched_outputs == outputs)
