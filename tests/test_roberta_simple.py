import torch
import torch.nn as nn
import torch.optim as optim
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from peft import LoraConfig, get_peft_model, TaskType
import logging

logging.basicConfig(
    filename="training_log.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def tokenize_function(examples, tokenizer, max_length=128):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "FacebookAI/xlm-roberta-large"

    dataset = load_dataset("glue", "sst2")

    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

    tokenized_datasets = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, max_length=128),
        batched=True
    )
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # train_loader = DataLoader([train_dataset[i] for i in range(64)], batch_size=64, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=64)
    eval_loader = DataLoader(eval_dataset, batch_size=128)

    model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)


    lora_config = LoraConfig(
        # task_type=TaskType.SEQ_CLS,
        # inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            'query',
            'key',
            'value',
        ],
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.to(device)

    model.print_trainable_parameters()
    # ----------------------------------------------------------------------

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    train_accuracy_metric = Accuracy(task="multiclass", num_classes=2).to(device)
    eval_accuracy_metric = Accuracy(task="multiclass", num_classes=2).to(device)

    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_accuracy_metric.reset()
        logging.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()

            if (step + 1) % 50 == 1 and False:
                scaler.unscale_(optimizer)
                # print(f"[Debug] Step {step+1}, loss={loss.item():.4f}")
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        # print(f"   Param: {name}, grad.shape={list(param.grad.shape)}, grad.norm={grad_norm:.6f}")

                for name, param in model.named_parameters():
                    if 'lora' in name:
                        print(param)
                        print('-' * 100)
                        print(param.grad)
                        break

                # input('> ')

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            train_accuracy_metric.update(predictions, batch["labels"])

            if (step + 1) % 50 == 0:
                avg_loss = running_loss / (step + 1)
                current_acc = train_accuracy_metric.compute().item()
                print(f"Epoch {epoch + 1}, Step {step + 1}/{len(train_loader)}, "
                      f"Loss: {avg_loss:.4f}, Train Acc: {current_acc:.4f}")

        avg_train_loss = running_loss / len(train_loader)
        train_acc = train_accuracy_metric.compute().item()
        print(f"[Epoch {epoch + 1}] avg training loss: {avg_train_loss:.4f}, accuracy: {train_acc:.4f}")

        model.eval()
        eval_loss = 0.0
        eval_accuracy_metric.reset()
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                eval_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                eval_accuracy_metric.update(predictions, batch["labels"])

        avg_eval_loss = eval_loss / len(eval_loader)
        eval_acc = eval_accuracy_metric.compute().item()
        print(f"[Epoch {epoch+1}] eval loss: {avg_eval_loss:.4f}, eval accuracy: {eval_acc:.4f}")
        ''''''


if __name__ == "__main__":
    main()
