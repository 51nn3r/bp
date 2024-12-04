import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim


class TestNN(nn.Module):
    def __init__(self, inp_dim, out_dim, **kwargs):
        super().__init__(**kwargs)

        self._inp_dim = inp_dim
        self._out_dim = out_dim
        self._layers = [nn.Linear(inp_dim, out_dim) for _ in range(4)]
        for idx, l in enumerate(self._layers):
            self.register_module(f'l{idx}', l)

    def forward(self, inputs):
        x = inputs
        for layer in self._layers:
            x = layer(x)
            x = torch.relu(x)
        return x


batch_size = 10
input_dim = 32
output_dim = 32

train_dataset_size = 1000

src_data = torch.rand((train_dataset_size, input_dim))
tgt_data = torch.rand((train_dataset_size, output_dim))

dataset = TensorDataset(src_data, tgt_data)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = TestNN(input_dim, output_dim)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs1, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')
