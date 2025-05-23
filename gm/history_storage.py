from typing import List

import torch


class HistoryStorage:
    _inputs: List[torch.Tensor]
    _keys: List[torch.Tensor]
    _outputs: List[torch.Tensor]
    _modes: List[torch.Tensor]
    _tactic_fillings_list: List[torch.Tensor]
    _strategy_fillings_list: List[torch.Tensor]

    def __init__(self):
        self.reset()

    def reset(self):
        self._inputs = []
        self._keys = []
        self._outputs = []
        self._modes = []
        self._tactic_fillings_list = []
        self._strategy_fillings_list = []

    def store_step_data(
            self,
            inp: torch.Tensor,
            key: torch.Tensor,
            output: torch.Tensor,
            mode: torch.Tensor,
            tactic_fillings: torch.Tensor,
            strategy_fillings: torch.Tensor,
    ):
        self._inputs.append(inp)
        self._keys.append(key)
        self._outputs.append(output)
        self._modes.append(mode)
        self._tactic_fillings_list.append(tactic_fillings)
        self._strategy_fillings_list.append(strategy_fillings)

    def restore_step_data(
            self,
            step_index: int,
    ):
        if step_index >= self.history_size:
            raise f"step_index ({step_index}) must be < history size ({self.history_size})"

        inp = self._inputs[step_index]
        keys_history = self.keys_history[:step_index]
        key = self.keys_history[step_index]
        output = self._outputs[step_index]
        mode = self._modes[step_index]
        tactic_fillings_history = self._tactic_fillings_list[:step_index]
        tactic_fillings = self._tactic_fillings_list[step_index]
        strategy_fillings = self._strategy_fillings_list[step_index]

        return (
            inp,
            keys_history,
            key,
            output,
            mode,
            tactic_fillings_history,
            tactic_fillings,
            strategy_fillings,
        )

    @property
    def history_size(self):
        return len(self.keys_history)

    @property
    def inputs_history(self):
        return self._inputs.copy()

    @property
    def keys_history(self):
        return self._keys.copy()

    @property
    def outputs_history(self):
        return self._outputs.copy()

    @property
    def tactic_fillings_history(self):
        return self._tactic_fillings_list.copy()
