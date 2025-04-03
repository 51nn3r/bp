from dataclasses import dataclass

from gm.layers.weights_storage.configs.weights_storage_config import WeightsStorageConfig
from gm.lora.enable_strategy.base_lora_enable_strategy import BaseLoraEnableStrategy
from gm.lora.enable_strategy.lora_enable_all_strategy import LoraEnableAllStrategy
from gm.lora.init_strategy.base_lora_init_strategy import BaseLoRAInitStrategy
from gm.lora.init_strategy.lora_full_init_strategy import LoRAFullInitStrategy


@dataclass
class LoraWeightsStorageConfig(WeightsStorageConfig):
    lora_init_strategy: BaseLoRAInitStrategy = LoRAFullInitStrategy
    lora_enable_strategy_cls: type(BaseLoraEnableStrategy) = LoraEnableAllStrategy

    enabled_adapters_proportion: float = 1 / 3  # used in XgbLoraEnableStrategy
