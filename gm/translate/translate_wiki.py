import json
from datasets import load_dataset
import tiktoken

from gm.translate.translator import get_translator

translator = get_translator()


def translate_batch(texts: list[str], max_tokens: int) -> list[str]:
    """
    Translate a batch of texts and return list of translated strings.
    Assumes translator() can accept list inputs.
    """
    results = translator(texts, max_length=max_tokens)
    return [res['translation_text'] for res in results]


def process_wiki40b(total_samples: int = 200000,
                    max_tokens: int = 512,
                    batch_size: int = 8,
                    output_file: str = "wiki40b_translated.jsonl"):
    """
    Load Wiki40B, translate in batches, chunk into max_tokens, and save.
    """
    ds = load_dataset("google/wiki40b", "en", split="train")
    enc = tiktoken.get_encoding("gpt2")

    samples = []
    raw_texts = []
    count = 0

    for example in ds:
        raw_texts.append(example["text"])
        count += 1

        # When enough texts collected or end of dataset
        if len(raw_texts) >= batch_size or (not raw_texts and count < total_samples):
            # Translate batch
            translated_texts = translate_batch(raw_texts, max_tokens)
            raw_texts.clear()

            # Process each translated text
            for translated in translated_texts:
                token_ids = enc.encode(translated)
                for i in range(0, len(token_ids), max_tokens):
                    if count >= total_samples:
                        break

                    chunk_ids = token_ids[i: i + max_tokens]
                    chunk_text = enc.decode(chunk_ids)
                    samples.append({"text": chunk_text})

                if count % 1 == 0 and count > 0:
                    print(f'[*] Translated and chunked {count} / {total_samples}')

                if count >= total_samples:
                    break

        if count >= total_samples:
            break

    # Flush any remaining raw_texts
    if count < total_samples and raw_texts:
        translated_texts = translate_batch(raw_texts, max_tokens)
        for translated in translated_texts:
            token_ids = enc.encode(translated)
            for i in range(0, len(token_ids), max_tokens):
                if count >= total_samples:
                    break
                chunk_ids = token_ids[i: i + max_tokens]
                chunk_text = enc.decode(chunk_ids)
                samples.append({"text": chunk_text})
                count += 1
            if count >= total_samples:
                break

    # Save to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved {len(samples)} samples to {output_file}")


if __name__ == "__main__":
    process_wiki40b(100000, 1024, 10, 'wiki40b_translated.json')
