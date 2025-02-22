from abc import ABC, abstractmethod
from typing import List, Dict, Any


class AbstractParsingStrategy(ABC):
    _required_keys: Dict[str, Any]
    _use_cache: bool
    _cached_positions: Dict[str: int] | None

    def __init__(
            self,
            required_keys: Dict[str, Any],
            use_cache: bool = True,
    ):
        self._required_keys = required_keys
        self._use_cache = use_cache
        self._cached_kwargs_positions = None

    @abstractmethod
    def parse(
            self,
            *args,
            **kwargs,
    ):
        pass

    def _parse_args(
            self,
            args,
            kwargs,
    ):
        result_kwargs = {}

        if self._use_cache and self._cached_positions is not None:
            for key, idx in self._cached_positions.items():
                result_kwargs[key] = kwargs.get(key, args[idx])

            return args[len(self._cached_positions):], result_kwargs

        new_cached_positions = {} if self._use_cache else None
        remaining_args = list(args)

        for key in self._required_keys:
            if key in kwargs:
                result_kwargs[key] = kwargs[key]
            elif remaining_args:
                result_kwargs[key] = remaining_args.pop(0)
                if self._use_cache:
                    new_cached_positions[key] = len(args) - len(remaining_args) - 1
            else:
                raise ValueError(f"Missing required argument: {key}")

        if self._use_cache:
            self._cached_positions = new_cached_positions

        return remaining_args, result_kwargs
