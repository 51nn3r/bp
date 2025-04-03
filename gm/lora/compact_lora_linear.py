from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, r=4, alpha=1.0, lora_enabled=True):
        """
        Слой, который действует как nn.Linear, но с добавлением LoRA-поправки.

        Параметры:
          in_features: размер входного вектора.
          out_features: размер выходного вектора.
          bias: использовать ли смещение.
          r: ранк LoRA (число столбцов в матрице lora_A и строк в lora_B).
          alpha: масштабирующий коэффициент для LoRA.
          lora_enabled: если True, LoRA поправки применяются (по умолчанию True).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.lora_enabled = lora_enabled

        # Основной линейный слой
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Если r > 0, инициализируем LoRA-параметры:
        if r > 0:
            # Для корректного умножения в PyTorch, где линейный слой использует вес shape (out_features, in_features):
            # Пусть lora_A имеет форму (r, in_features) и lora_B — (out_features, r).
            # Тогда произведение lora_B @ lora_A имеет форму (out_features, in_features).
            self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.randn(out_features, r) * 0.01)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        # Вычисляем стандартное линейное преобразование
        output = self.linear(x)

        if self.lora_enabled and self.r > 0:
            # Вычисляем LoRA-поправку:
            # lora_update = lora_B @ lora_A имеет форму (out_features, in_features)
            # F.linear(x, weight) ожидает weight shape (out_features, in_features)
            lora_update = F.linear(x, self.lora_B @ self.lora_A)
            # Масштабируем поправку, обычно коэффициент alpha делится на r
            output = output + (self.alpha / self.r) * lora_update

        return output

    @staticmethod
    def from_module(
            module: nn.Linear,
            r: int = 4,
            alpha: float = 1.,
    ) -> CompactLoRALinear:
        """Extracts the required parameters from the source module and creates a pseudo-layer instance"""
        linear = CompactLoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            r=r,
            alpha=alpha,
        )

        return linear


def patch_module(module: nn.Module, target_modules):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and (target_modules is None or name in target_modules):
            pseudo_module = CompactLoRALinear.from_module(
                module=child,
            )
            setattr(module, name, pseudo_module)
        else:
            patch_module(child, target_modules)
