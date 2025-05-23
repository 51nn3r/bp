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

from gm.utils.metrics import masked_nll

from gm.settings import CPU_DEVICE, CUDA_DEVICE


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
        self._weights_storage.enable_grad()

        return self

    def fit(
            self,
            train_dataset,
            test_dataset=None,
            num_epochs=3,
            batch_size=4,
            lr=1e-4,
            k=1,
            device=None,
    ):
        if device is None:
            device = torch.device(CPU_DEVICE)

        # ---------- dataloader helper ----------
        def collate_fn(examples):
            return {
                "input_ids": torch.stack([e["input_ids"] for e in examples]),
                "attention_mask": torch.stack([e["attention_mask"] for e in examples]),
                "labels": torch.stack([e["labels"] for e in examples]),
            }

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # ---------- move model ----------
        self._module.to(device)

        # Freeze all parameters first
        for p in self.parameters():
            p.requires_grad = False

        self.train()

        # Collect trainable LoRA parameters
        lora_params = []
        for layer_adapters in self._weights_storage._lora_modules:
            for adapter in layer_adapters or []:
                if adapter is not None:
                    for p in adapter.parameters():
                        p.requires_grad = True
                        lora_params.append(p)

        optimizer = Adam8bit(
            [{"params": lora_params, "lr": lr, "weight_decay": 0.01}],
        )
        scaler = GradScaler()

        steps_per_epoch = len(train_loader)
        update_freq = num_epochs * steps_per_epoch // k  # gradient-accumulation steps
        grad_accumulation = 8
        print(f"[+] gradient accumulation steps count = {update_freq}")

        for epoch in range(num_epochs):
            self.train()
            self._weights_storage.enable_grad()

            running_nll = 0.0
            running_loss = 0.0
            token_count = 0

            for step, batch in enumerate(train_loader, start=1):
                # ---------- batch to device ----------
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                mask = batch.get("attention_mask")
                if mask is None:
                    mask = (inputs != 0).long()

                mask = mask.to(device)

                # ---------- forward / backward ----------
                with autocast(dtype=torch.bfloat16):
                    # Let GPT-Neo create its own causal mask; disable cache
                    outputs = self._module(
                        inputs,
                        use_cache=False,
                        labels=None,  # we do the loss ourselves
                    )
                    logits = outputs.logits
                    nll_loss = masked_nll(logits, labels, mask)  # mask = padding mask you built
                    loss = nll_loss / grad_accumulation

                running_loss += loss.item()
                running_nll += nll_loss.item()
                token_count += mask.sum().item()

                scaler.scale(loss).backward()

                # optimizer step every update_freq mini-batches
                if step % update_freq == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    # ---------- XGBLoRA boost step ----------
                    self._weights_storage.apply_weights()
                    self._weights_storage.reset_lora()
                    # reset Adam moments for fresh LoRA phase
                    for p in lora_params:
                        st = optimizer.state.get(p)
                        if st:
                            st.get("exp_avg", torch.zeros(1)).zero_()
                            st.get("exp_avg_sq", torch.zeros(1)).zero_()

                # ---------- logging ----------
                if (step + 1) % 100 == 0:
                    avg_nll = running_nll / step
                    ppl = float(torch.exp(torch.tensor(avg_nll)))
                    print(
                        f"[Epoch {epoch + 1}] step {step + 1}/{len(train_loader)} | loss={running_loss / (step + 1)} | "
                        f"avg NLL/token={avg_nll:.4f} | PPL={ppl:.2f}")

            # ---------- epoch summary ----------
            mean_nll = running_nll / len(train_loader)
            mean_ppl = float(torch.exp(torch.tensor(mean_nll)))
            print(f"[Epoch {epoch + 1} END] mean NLL/token={mean_nll:.4f} | PPL={mean_ppl:.2f}")

            # ---------- evaluation ----------
            if test_loader is not None:
                self.eval()
                total_nll = 0.0
                total_tokens = 0
                with torch.no_grad():
                    for batch in test_loader:
                        inputs = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        mask = batch.get("attention_mask")
                        if mask is None:
                            mask = (inputs != 0).long()
                        mask = mask.to(device)

                        with autocast():
                            logits = self._module(inputs, attention_mask=mask, labels=None).logits
                            nll = masked_nll(logits, labels, mask)

                        total_nll += nll.item() * mask.sum().item()
                        total_tokens += mask.sum().item()

                eval_nll = total_nll / total_tokens
                eval_ppl = float(torch.exp(torch.tensor(eval_nll)))
                print(f"[Epoch {epoch + 1} EVAL] NLL/token={eval_nll:.4f} | PPL={eval_ppl:.2f}")

        # ---------- save final model ----------
        self.save_model(f"model_fin_{int(time()):d}.pth")

    def save_model(
            self,
            path='model.pth'
    ):
        torch.save(self._module.state_dict(), path)

    def load_model(
            self,
            path='model.pth',
            device_name=CUDA_DEVICE,  # TODO: change device_name -> self._device
    ):
        return self._module.load_state_dict(torch.load(path, map_location=torch.device(device_name)))

    def reset_parameters(self):
        self._weights_storage.reset_parameters()
