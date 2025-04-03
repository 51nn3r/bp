from abc import ABC, abstractmethod


class BaseLoraEnableStrategy(ABC):
    _adapters_count: int

    def __init__(
            self,
            adapters_count: int,
            *args,
            **kwargs,
    ):
        self._adapters_count = adapters_count

    @abstractmethod
    def get_activation_mask(
            self,
            *args,
            **kwargs,
    ):
        pass
