from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

from gm.layers.weights_storage import WeightsStorage
from gm.layers.shaped_layer import ShapedLayer

from gm.layers.pseudo_layers.pseudo_linear import PseudoLinear

PRINT_GRAD = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class TestShapedLayer(ShapedLayer):
    _shapes: List[torch.Size]

    def __init__(
            self,
            shapes: List[torch.Size | List],
            **kwargs,
    ):
        super().__init__(**kwargs)

        self._shapes = shapes

    @property
    def shapes(self) -> List[torch.Size]:
        return self._shapes


layers = [TestShapedLayer([torch.Size([2, 3]), torch.Size([3, 4])]) for _ in range(10)]

groups_count = 4
storage_size = 16


class TestNN(nn.Module):
    def __init__(self, inp_dim, out_dim, **kwargs):
        super().__init__(**kwargs)

        self._inp_dim = inp_dim
        self._out_dim = out_dim
        self._weights_storage = WeightsStorage(groups_count, storage_size, device)
        self._layers = [PseudoLinear(self._weights_storage, inp_dim, out_dim) for _ in range(16)]
        self._weights_storage.build_storage()

    def forward(self, selector, inputs):
        x = inputs
        for layer in self._layers:
            x = layer(selector, x)
            x = torch.relu(x)

        return x


batch_size = 10
input_dim = 256
output_dim = 256

train_dataset_size = 1000

src_data = torch.rand((train_dataset_size, input_dim,))
tgt_data = torch.rand((train_dataset_size, output_dim,))
selector = torch.randint(0, storage_size, (train_dataset_size, groups_count))

dataset = TensorDataset(src_data, selector, tgt_data)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = TestNN(input_dim, output_dim)

model = model.to(device)
print(sum(p.numel() for p in model.parameters()))

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs1, inputs2, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        targets = targets.to(device)
        outputs = model(inputs2, inputs1)
        loss = criterion(outputs, targets)
        loss.backward()
        if PRINT_GRAD:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f'{name} - grad: {torch.sum(param.grad, dim=(-1, -2))}')

        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')
