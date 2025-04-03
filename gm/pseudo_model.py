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
from torch import optim
from bitsandbytes.optim import Adam8bit

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
            target_modules: List[str] | None = None,
    ):
        for name, child in module.named_children():
            if child.__class__ in mapping and (
                    target_modules is None
                    or any(t in name for t in target_modules)
            ):
                pseudo_cls = mapping[child.__class__]
                pseudo_module = pseudo_cls.from_module(
                    weights_storage=self._weights_storage,
                    module=child,
                )
                setattr(module, name, pseudo_module)
            else:
                self._patch_module(child, mapping, target_modules)

    @staticmethod
    def create_patched_pseudo_model(
            weights_storage: WeightsStorage,
            module: nn.Module,
            mapping: Dict[Type[nn.Module], Type[PseudoLayer]],
            target_modules: List[str] | None = None,
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

    def fit(self, train_dataset, test_dataset=None, num_epochs=3, batch_size=4, lr=1e-4, device=None):
        if device is None:
            device = torch.device(CPU_DEVICE)

        def collate_fn(examples):
            return {
                "input_ids": torch.stack([torch.tensor(e["input_ids"]) for e in examples]),
                "attention_mask": torch.stack([torch.tensor(e["attention_mask"]) for e in examples]),
                "labels": torch.stack([torch.tensor(e["labels"]) for e in examples]),
            }

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        self._module.to(device)
        for param in self.parameters():
            param.requires_grad = False
            pass

        for param in self._module.classifier.parameters():
            param.requires_grad = True

        self.train()

        # optimizer = Adam8bit([p for p in self._module.parameters() if p.requires_grad], lr=lr)
        optimizer = optim.AdamW([p for p in self._module.parameters() if p.requires_grad], lr=lr)

        start_step_time = time()
        batches_count = len(train_loader)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for step, batch in enumerate(train_loader):
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                if 'attention_mask' in batch:
                    attention_mask = batch["attention_mask"].to(device)
                else:
                    attention_mask = None

                optimizer.zero_grad()

                with autocast(dtype=torch.bfloat16):
                    if attention_mask is not None:
                        outputs = self._module(inputs, attention_mask=attention_mask,
                                               labels=labels)
                    else:
                        outputs = self._module(inputs, labels=labels)

                    loss = outputs.loss

                loss.backward(retain_graph=True)
                total_params_count = 0
                for param in self._module.parameters():
                    if param.grad is not None:
                        total_params_count += param.grad.numel()

                    pass

                optimizer.step()

                running_loss += loss.item()
                if step % 50 == 0:
                    current_time = time()
                    '''
                    print(f"Epoch {epoch + 1}, Step {step} / {batches_count}, Loss: {running_loss / (step + 1):.4f}, "
                          f"Time: {current_time - start_step_time}")
                    '''

                    print(f"Epoch {epoch + 1}, Step {step} / {batches_count}, Loss: {running_loss / 50}, "
                          f"Time: {current_time - start_step_time}")

                    start_step_time = current_time
                    running_loss = 0

                if step % 500 == 0:
                    # self._weights_storage.update_weights_and_reinit_lora()
                    pass

                if step % 250 == 0:
                    if test_dataset is not None:
                        self.eval()
                        self._weights_storage.enable_lora()
                        test_loss = 0.0
                        correct = 0
                        total = 0
                        test_batches = len(test_loader)
                        with torch.no_grad():
                            for t_step, t_batch in enumerate(test_loader):
                                t_inputs = t_batch["input_ids"].to(device)
                                t_labels = t_batch["labels"].to(device)
                                t_attention_mask = t_batch.get("attention_mask", None)
                                if t_attention_mask is not None:
                                    t_attention_mask = t_attention_mask.to(device)

                                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                                    t_outputs = self._module(t_inputs, attention_mask=t_attention_mask, labels=t_labels)
                                    t_loss = t_outputs.loss
                                    logits = t_outputs.logits

                                test_loss += t_loss.item()

                                predictions = torch.argmax(logits, dim=-1)
                                correct += (predictions == t_labels).sum().item()
                                total += t_labels.numel()

                        avg_test_loss = test_loss / test_batches
                        accuracy = correct / total if total > 0 else 0.0
                        print(
                            f"Epoch {epoch + 1} evaluation: avg test loss: {avg_test_loss:.4f}, accuracy: {accuracy:.4f}")

                        self.train()

            print(f"Epoch {epoch + 1} finished, avg loss: {running_loss / len(train_loader):.4f}")
            self.save_model(f'neo{epoch}_{int(time())}.pth')

            if test_dataset is not None:
                self.eval()
                test_loss = 0.0
                test_batches = len(test_loader)
                with torch.no_grad():
                    for t_step, t_batch in enumerate(test_loader):
                        t_inputs = t_batch["input_ids"].to(device)
                        t_labels = t_batch["labels"].to(device)
                        t_attention_mask = t_batch.get("attention_mask", None)
                        if t_attention_mask is not None:
                            t_attention_mask = t_attention_mask.to(device)

                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            t_outputs = self._module(t_inputs, attention_mask=t_attention_mask, labels=t_labels)
                            t_loss = t_outputs.loss
                        test_loss += t_loss.item()

                avg_test_loss = test_loss / test_batches
                print(f"Epoch {epoch + 1} evaluation: avg test loss: {avg_test_loss:.4f}")
                self.train()

    def save_model(
            self,
            path='model.pth'
    ):
        torch.save(self._module.state_dict(), path)

    def load_model(
            self,
            path='model.pth',
            device_name=CPU_DEVICE,  # TODO: change device_name -> self._device
    ):
        self._module.load_state_dict(torch.load(path, map_location=torch.device(device_name)))

    def reset_parameters(self):
        self._weights_storage.reset_parameters()
