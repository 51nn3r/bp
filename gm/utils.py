import torch

"""  Module utils  """


def move_to_device(
        data,
        device: torch._C.device,
):
    if isinstance(data, dict):
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(x, device) for x in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        raise TypeError("Unsupported data type")


"""  Dataset utils  """


def create_glue_dataset_neo(dataset, task, tokenizer, sep_tokens, max_length=128, mask_prompt: bool = True, ):
    sep_token_ids = [tokenizer.convert_tokens_to_ids(sep_token) for sep_token in sep_tokens]

    def tokenize_example(ex):
        if task.lower() in ["cola", "sst2"]:
            if len(sep_tokens) < 1:
                raise "Not enough separation tokens (needs 1)"

            tokenized = tokenizer(
                [s + sep_tokens[0] + str(l) for s, l in zip(ex["sentence"], ex["label"], )],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        elif task.lower() in ["mrpc", "qqp"]:
            if len(sep_tokens) < 2:
                raise "Not enough separation tokens (needs 2)"

            tokenized = tokenizer(
                ex["sentence1"] + sep_tokens[0] + ex["sentence2"] + sep_tokens[1],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        elif task.lower() == "mnli":
            if len(sep_tokens) < 2:
                raise "Not enough separation tokens (needs 2)"

            tokenized = tokenizer(
                ex["premise"] + sep_tokens[0] + ex["hypothesis"] + sep_tokens[1],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        else:
            if len(sep_tokens) < 1:
                raise "Not enough separation tokens (needs 1)"

            tokenized = tokenizer(
                ex.get("sentence", "") + sep_tokens[0],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )

        input_ids_batch = tokenized["input_ids"]
        attention_mask_batch = tokenized["attention_mask"]

        labels_batch = []
        for input_ids in input_ids_batch:
            labels = input_ids.copy()

            if mask_prompt:
                answer_start_position = input_ids.index(sep_token_ids[-1]) + 1
                if answer_start_position is None: answer_start_position = 0
                for i in range(answer_start_position):
                    labels[i] = -100

            labels_batch.append(labels)

        tokenized["labels"] = labels_batch
        tokenized["attention_mask"] = attention_mask_batch

        return tokenized

    dataset = dataset.map(tokenize_example, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return dataset


def create_glue_dataset(dataset, task, tokenizer, max_length=128):
    def tokenize_example(ex):
        if task.lower() in ["cola", "sst2"]:
            tokenized = tokenizer(
                ex["sentence"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        elif task.lower() in ["mrpc", "qqp"]:
            tokenized = tokenizer(
                ex["sentence1"],
                ex["sentence2"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        elif task.lower() == "mnli":
            tokenized = tokenizer(
                ex["premise"],
                ex["hypothesis"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        elif task.lower() == "qnli":
            tokenized = tokenizer(
                ex["question"],
                ex["sentence"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        else:
            tokenized = tokenizer(
                ex.get("sentence", ""),
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )

        tokenized["labels"] = ex["label"]
        return tokenized

    dataset = dataset.map(
        lambda ex: tokenize_example(ex),
        batched=True,
    )
    actual_columns_to_remove = [col for col in ["sentence", "question", "sentence1", "sentence2", "idx"] if col in dataset.column_names]
    dataset = dataset.remove_columns(actual_columns_to_remove)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset
