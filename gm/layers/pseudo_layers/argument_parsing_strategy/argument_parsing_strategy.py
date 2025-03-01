from typing import Tuple, Dict, Any


class ArgumentParsingStrategy:
    _required_keys: Dict[str, Any]
    _use_cache: bool
    _cached_positions: Dict[str, int] | None

    def __init__(
            self,
            required_keys: Dict[str, Any],
            use_cache: bool = True,
    ):
        self._required_keys = required_keys
        self._use_cache = use_cache
        self._cached_positions = None

    def parse(
            self,
            *args,
            **kwargs,
    ):
        return self._parse_args(args, kwargs)

    def _parse_args(
            self,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
    ) -> (Tuple[Any], Dict[str, Any]):

        if self._use_cache and self._cached_positions is not None:
            for key, idx in self._cached_positions.items():
                kwargs[key] = args[idx]

            return args[len(self._cached_positions):], kwargs

        new_cached_positions = {} if self._use_cache else None
        remaining_args = list(args)

        for key in self._required_keys:
            if key not in kwargs:
                if not remaining_args:
                    raise ValueError(f"Missing required argument: {key}")

                kwargs[key] = remaining_args.pop(0)
                if self._use_cache:
                    new_cached_positions[key] = len(args) - len(remaining_args) - 1

        if self._use_cache:
            self._cached_positions = new_cached_positions

        return remaining_args, kwargs
