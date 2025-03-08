from __future__ import annotations
from typing import Type, Dict, Mapping, Any, List
from time import time

import torch
from torch import nn
from torch.nn.functional import mse_loss, cross_entropy
import torch.nn.functional as F
from torch.nn.modules.module import T
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, Adam

from gm.layers.weights_storage.weights_storage import WeightsStorage
from gm.layers.pseudo_layers.pseudo_layer import PseudoLayer

from gm.settings import CPU_DEVICE


class PseudoModule(nn.Module):
    _weights_storage: WeightsStorage
    _module: nn.Module

    def __init__(
            self,
            weights_storage: WeightsStorage,
            module: nn.Module,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._weights_storage = weights_storage
        self._module = module

    def _patch_module(
            self,
            module: nn.Module,
            mapping: Dict[Type[nn.Module], Type[PseudoLayer]],
            target_modules: List[str],
    ):
        for name, child in module.named_children():
            if child.__class__ in mapping and name in target_modules:
                pseudo_cls = mapping[child.__class__]
                pseudo_module = pseudo_cls.from_module(
                    weights_storage=self._weights_storage,
                    module=child,
                )
                setattr(module, name, pseudo_module)
            else:
                self._patch_module(child, mapping, target_modules)

    def _compute_loss(self, logits, labels, vocab_size, ignore_index=-100):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        # Сдвигаем логиты и метки:
        shift_logits = logits[:, :-1, :].contiguous()  # убираем последний токен
        shift_labels = labels[:, 1:].contiguous()  # убираем первый токен

        # Переводим в форму [N, vocab_size] и [N]:
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=ignore_index
        )
        return loss

    @staticmethod
    def create_patched_pseudo_model(
            weights_storage: WeightsStorage,
            module: nn.Module,
            mapping: Dict[Type[nn.Module], Type[PseudoLayer]],
            target_modules: List[str],
    ) -> PseudoModule:
        pseudo_model = PseudoModule(weights_storage, module)
        pseudo_model._patch_module(module, mapping, target_modules)

        return pseudo_model

    def eval(self: T) -> T:
        super().eval()
        self._weights_storage.eval()

        return self

    def train(self: T, mode: bool = True) -> T:
        super().train(mode)
        self._weights_storage.train()

        return self

    def fit(self, train_dataset, num_epochs=3, batch_size=4, lr=1e-4, device=None):
        if device is None:
            device = torch.device(CPU_DEVICE)

        # Создаем DataLoader с кастомным collate_fn для паддинга (предполагаем, что данные уже в виде словаря с "inputs" и "labels")
        def collate_fn(examples):
            # Пример: если ваши данные уже содержат тензоры, можно использовать torch.stack
            # или, если это списки, вызвать tokenizer.pad(...) если он доступен
            return {
                "input_ids": torch.stack([torch.tensor(e["input_ids"]) for e in examples]),
                "labels": torch.stack([torch.tensor(e["labels"]) for e in examples])
            }

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        self._module.to(device)
        for param in self.parameters():
            param.requires_grad = False
            pass

        # param.requires_grad = True
        self.train()

        # Используем 8-бит Adam:
        optimizer = Adam([p for p in self._module.parameters() if p.requires_grad], lr=lr)

        start_step_time = time()
        batches_count = len(train_loader)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for step, batch in enumerate(train_loader):
                # Предполагаем, что batch имеет ключи "inputs" и "labels"
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                if 'attention_mask' in batch:
                    attention_mask = batch["attention_mask"].to(device)
                else:
                    attention_mask = None

                optimizer.zero_grad()

                # Используем autocast для BF16 (если ваша платформа поддерживает BF16)
                with autocast(dtype=torch.bfloat16):
                    if attention_mask is not None:
                        outputs = self._module(inputs, attention_mask=attention_mask,
                                               labels=labels)  # forward возвращает логиты
                    else:
                        outputs = self._module(inputs, labels=labels)

                    # loss = self._compute_loss(logits, labels, vocab_size)
                    loss = outputs.loss

                loss.backward(retain_graph=True)
                total_params_count = 0
                for param in self._module.parameters():
                    if param.grad is not None:
                        total_params_count += param.grad.numel()

                    pass

                optimizer.step()

                running_loss += loss.item()
                if step % 500 == 0:
                    current_time = time()
                    print(f"Epoch {epoch + 1}, Step {step} / {batches_count}, Loss: {running_loss / (step + 1):.4f}, "
                          f"Time: {current_time - start_step_time}")

                    start_step_time = current_time

                if step % 1000 == 0:
                    self._weights_storage.update_weights_and_reinit_lora()
                    pass

            print(f"Epoch {epoch + 1} finished, avg loss: {running_loss / len(train_loader):.4f}")
            self.save_model(f'neo{epoch}.pth')
            # self.save_model(f'/content/drive/MyDrive/Colab\ Notebooks/bp/models/neo{epoch}.pth')

    def save_model(
            self,
            path='model.pth'
    ):
        torch.save(self._module.state_dict(), path)

    def load_model(
            self,
            path='model.pth'
    ):
        self._module.load_state_dict(torch.load(path))

    def reset_parameters(self):
        self._weights_storage.reset_parameters()
