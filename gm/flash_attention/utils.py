from typing import List, Dict, Type

from torch import nn

from gm.flash_attention.XformersFlashMHA import XformersFlashMHA


def patch_model_to_flash_mha(
        self,
        module: nn.Module,
        mapping: Dict[Type[nn.Module], Type[nn.Module]],
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
