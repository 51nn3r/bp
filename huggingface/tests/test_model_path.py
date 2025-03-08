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
        # Линейные слои, которые предполагается патчить
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)

        # Разные активации
        self.act_relu = nn.ReLU()
        self.act_tanh = nn.Tanh()
        self.act_sigmoid = nn.Sigmoid()

        # Дополнительные слои, которые не должны патчиться
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(hidden_dim)

        # Вложенный подмодуль с несколькими Linear слоями и активацией
        self.submodule = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, False)
        )

    def forward(self, x, small_test=False):
        if small_test:
            return self.linear(x)

        # Первый блок: линейный слой + активация
        x = self.linear1(x)
        x = self.act_relu(x)
        x = self.bn(x)

        # Второй блок: второй линейный слой + другая активация + dropout
        x = self.linear2(x)
        x = self.act_tanh(x)
        x = self.dropout(x)

        # Третий блок: третий линейный слой + активация
        x = self.linear3(x)
        x = self.act_sigmoid(x)

        # Пропуск через вложенный подмодуль, где также присутствуют Linear слои
        x = self.submodule(x)
        return x


if __name__ == '__main__':
    # Пример инициализации модели и прохождения через нее случайного входа
    model = TestModel(input_dim=10, hidden_dim=32, output_dim=5)
    print(model)

    model.eval()
    dummy_input = torch.randn(4, 10)

    linear_weights = [model.linear.weight, model.linear.bias]
    print('real weights', linear_weights)
    print('=' * 100)

    output = model(dummy_input, False)
    print("Output1 (must be equal to Output2):", output)
    output = model(dummy_input, False)
    print("Output2 (must be equal to Output1):", output)

    init_strategy = LoRAFullInitStrategy(LoRA)
    weights_storage = LoRAWeightsStorage(ArgumentParsingStrategy({}), init_strategy)

    print(model.linear1.bias)
    pseudo_model = PseudoModule.create_patched_pseudo_model(
        weights_storage=weights_storage,
        module=model,
        mapping={nn.Linear: PseudoLinear},
    )
    weights_storage.build_storage()
    for layer_lora in weights_storage._lora_modules:
        for lora in layer_lora:
            print(lora._shape)

    print('-' * 100)
    pseudo_weights = weights_storage.forward(0)

    patched_output = model(dummy_input, False)
    print("Patched output:", patched_output)

    print('=' * 100)
    print('pseudo weights', pseudo_weights)

    print('=' * 200)

    dummy_input_ids = torch.randn(1000, 10)
    dummy_labels = torch.randn(1000, 5)
    train_ds = Dataset.from_dict({
        'input_ids': dummy_input_ids,
        'labels': dummy_labels,
    })

    pseudo_model.fit(
        train_dataset=train_ds,
        vocab_size=128008,
        batch_size=4,
        lr=1e-4,
        num_epochs=3,
    )
