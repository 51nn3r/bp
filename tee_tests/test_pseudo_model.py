import unittest
import torch
from torch import nn
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time

from gm.layers.pseudo_layers.argument_parsing_strategy.argument_parsing_strategy import ArgumentParsingStrategy
from gm.layers.pseudo_layers.pseudo_linear import PseudoLinear
from gm.layers.weights_storage.lora_weights_storage import LoRAWeightsStorage
from gm.layers.weights_storage.weights_storage import WeightsStorage
from gm.lora.init_strategy.lora_full_init_strategy import LoRAFullInitStrategy
from gm.lora.lora import LoRA
from gm.pseudo_model import PseudoModule

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class TestModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TestModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)

        self.act_relu = nn.ReLU()
        self.act_tanh = nn.Tanh()
        self.act_sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(hidden_dim)

        self.submodule = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, False)
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_relu(x)
        x = self.bn(x)

        x = self.linear2(x)
        x = self.act_tanh(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = self.act_sigmoid(x)

        x = self.submodule(x)
        return x


class TestPseudoModule(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.test_model = TestModel(10, 20, 10)
        self.weights_storage = LoRAWeightsStorage(ArgumentParsingStrategy({}), LoRAFullInitStrategy(LoRA),
                                                  device=device)

        self.pseudo_model = PseudoModule.create_patched_pseudo_model(
            weights_storage=self.weights_storage,
            module=self.test_model,
            mapping={nn.Linear: PseudoLinear},
        )

    def test_linear_layer_patching(self):
        self.assertIsInstance(self.test_model.linear1, PseudoLinear,
                              "Linear layer should be patched by PseudoLinear layer.")

    def test_weights_storage_initialization(self):
        def count_linear_layers(module: nn.Module):
            if isinstance(module, nn.Linear):
                return 1

            count = 0
            for child in module.children():
                count += count_linear_layers(child)

            return count

        ll_count = count_linear_layers(self.test_model)
        self.weights_storage.build_storage(rank=4)
        self.assertEqual(len(self.weights_storage._lora_modules), ll_count,
                         "Weights storage should contain exactly one LoRA layer.")

    def test_update_weights_and_reinit_lora(self):
        self.weights_storage.build_storage(rank=4)
        original_weights = [layer._matrices[0].clone() for layer in self.weights_storage.lora_layers]
        self.weights_storage.update_weights_and_reinit_lora()
        updated_weights = [layer._matrices[0] for layer in self.weights_storage.lora_layers]

        for original, updated in zip(original_weights, updated_weights):
            self.assertFalse(torch.equal(original, updated),
                             "LoRA weights should change after update and reinitialization.")

    def test_eval_mode_disables_grad(self):
        self.test_model.eval()
        for param in self.test_model.parameters():
            self.assertFalse(param.requires_grad, "Parameters should not require grad in eval mode.")

    def test_train_mode_activation(self):
        self.test_model.train()
        for param in self.test_model.parameters():
            self.assertTrue(param.requires_grad, "Parameters should require grad in train mode.")


if __name__ == '__main__':
    unittest.main()
