from typing import List

import itertools

import torch
from torch import nn
from torch import optim

from gm.history_storage import HistoryStorage


class FillingsBasedModel(nn.Module):
    _key_gen: nn.Module
    _handler: nn.Module
    _tactic_fillings_gen: nn.Module
    _strategy_fillings_gen: nn.Module
    _mode_gen: nn.Module

    _tactic_optimizer: optim.Optimizer
    _strategy_optimizer: optim.Optimizer
    _fillings_optimizer: optim.Optimizer

    _criterion: nn.Module

    _history_storage: HistoryStorage

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        self._key_gen = ...
        self._handler = ...
        self._tactic_fillings_gen = ...
        self._strategy_fillings_gen = ...
        self._mode_gen = ...

        self._tactic_optimizer = optim.Adam(self._handler.parameters())
        self._strategy_optimizer = optim.Adam(itertools.chain(
            self._key_gen.parameters(),
            self._mode_gen.parameters(),
        ))
        self._fillings_optimizer = optim.Adam(itertools.chain(
            self._tactic_fillings_gen.parameters(),
            self._strategy_fillings_gen.parameters(),
        ))

        self._history_storage = HistoryStorage()

    def forward(
            self,
            inp: torch.Tensor,
            keys_history: List[torch.Tensor],
            tactic_fillings_history: List[torch.Tensor],
    ):
        keys_history = torch.cat(keys_history, dim=-3)
        key = self._key_gen(inp, keys_history)
        output = self._handler(key, inp)
        mode = self._mode_gen(key, inp)
        tactic_fillings = self._tactic_fillings_gen(key, inp, output)
        tactic_fillings_history.append(tactic_fillings)
        tactic_fillings_history = torch.cat(tactic_fillings_history, dim=-3)
        strategy_fillings = self._strategy_fillings_gen(key, keys_history, tactic_fillings_history)
        return key, output, mode, tactic_fillings, strategy_fillings

    def train_step(
            self,
            inp: torch.Tensor,
            save_step=True,
    ):
        self._tactic_optimizer.zero_grad()
        self._strategy_optimizer.zero_grad()
        key, output, mode, tactic_fillings, strategy_fillings = self.forward(
            inp=inp,
            keys_history=self._history_storage.keys_history,
            tactic_fillings_history=self._history_storage.tactic_fillings_history,
        )
        y_tactic_fillings = torch.zeros(strategy_fillings.shape)
        tactic_loss = self._criterion(strategy_fillings, y_tactic_fillings)
        tactic_loss.backward()
        self._tactic_optimizer.step()

        y_strategy_fillings = torch.zeros(strategy_fillings.shape)
        strategy_loss = self._criterion(strategy_fillings, y_strategy_fillings)
        strategy_loss.backward()
        self._strategy_optimizer.step()

        if save_step is True:
            self._history_storage.store_step_data(
                inp=inp,
                key=key,
                output=output,
                mode=mode,
                strategy_fillings=strategy_fillings,
                tactic_fillings=tactic_fillings,
            )

        return strategy_loss, tactic_loss

    def fillings_train_step(
            self,
            y_tactic_fillings: torch.Tensor,
            y_strategy_fillings: torch.Tensor,
    ):
        self._fillings_optimizer.zero_grad()
        tactic_fillings_list: List[torch.Tensor] = []
        strategy_fillings_list: List[torch.Tensor] = []
        for step_index in range(self._history_storage.history_size):
            (
                inp,
                keys_history,
                key,
                output,
                mode,
                tactic_fillings_history,
                _,
                _,
            ) = self._history_storage.restore_step_data(step_index)

            tactic_fillings = self._tactic_fillings_gen(
                key,
                inp,
                output
            )
            tactic_fillings_list.append(tactic_fillings)
            strategy_fillings = self._strategy_fillings_gen(
                key,
                keys_history,
                torch.cat([
                    tactic_fillings_history,
                    tactic_fillings,
                ], dim=-3)
            )
            strategy_fillings_list.append(strategy_fillings)

        # sum fillings
        tactic_fillings_sum = torch.sum(torch.stack(tactic_fillings_list), dim=0)
        strategy_fillings_sum = torch.sum(torch.stack(strategy_fillings_list), dim=0)

        # compute loss
        tactic_loss = self._criterion(tactic_fillings_sum, y_tactic_fillings)
        strategy_loss = self._criterion(strategy_fillings_sum, y_strategy_fillings)

        # apply gradients
        tactic_loss.backward()
        strategy_loss.backward()
        self._fillings_optimizer.step()

        return tactic_loss, strategy_loss
