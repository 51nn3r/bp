import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class CustomTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 max_seq_length, num_classes):
        super(CustomTransformer, self).__init__()

        self.model_dim = model_dim

        # Эмбеддинг для входов
        self.embedding = nn.Embedding(input_dim, model_dim)

        # Позиционный эмбеддинг
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, model_dim))

        # Энкодер
        encoder_layers = TransformerEncoderLayer(model_dim, num_heads, dim_feedforward)
        self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        # Декодер
        decoder_layers = TransformerDecoderLayer(model_dim, num_heads, dim_feedforward)
        self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

        # Выходной линейный слой
        self.fc_out = nn.Linear(model_dim, num_classes)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src_emb = self.embedding(src) * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        src_emb += self.positional_encoding[:, :src_emb.size(1), :]

        tgt_emb = self.embedding(tgt) * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        tgt_emb += self.positional_encoding[:, :tgt_emb.size(1), :]

        memory = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        output = self.fc_out(output)
        return output


# Пример использования кастомного трансформера
input_dim = 1000  # Размер словаря
model_dim = 128  # Размерность модели
num_heads = 8  # Количество голов в multi-head attention
num_encoder_layers = 6  # Количество слоев энкодера
num_decoder_layers = 6  # Количество слоев декодера
dim_feedforward = 128  # Размерность feedforward слоя
max_seq_length = 100  # Максимальная длина последовательности
num_classes = 10  # Количество классов (для классификации, например)
batch_size = 32
num_epochs = 10
learning_rate = 1e-3

model = CustomTransformer(input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward,
                          max_seq_length, num_classes)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Подготовка данных
# Пример: создание случайных данных для обучения и валидации
src_data = torch.randint(0, input_dim,
                         (1000, max_seq_length))  # 1000 примеров, последовательность длиной max_seq_length
tgt_data = torch.randint(0, input_dim, (1000, max_seq_length))
labels = torch.randint(0, num_classes,
                       (1000, max_seq_length))  # 1000 примеров, метки для каждой позиции в последовательности

dataset = TensorDataset(src_data, tgt_data, labels)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Функция для обучения модели на одной эпохе
def train_one_epoch(model, data_loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for src, tgt, label in data_loader:
        optimizer.zero_grad()
        output = model(src, tgt)

        # Преобразование размера выхода и меток для функции потерь
        output = output.view(-1,
                             num_classes)  # Из [batch_size, seq_length, num_classes] в [batch_size * seq_length, num_classes]
        label = label.view(-1)  # Из [batch_size, seq_length] в [batch_size * seq_length]

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


# Функция для валидации модели
def validate(model, data_loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt, label in data_loader:
            output = model(src, tgt)

            # Преобразование размера выхода и меток для функции потерь
            output = output.view(-1,
                                 num_classes)  # Из [batch_size, seq_length, num_classes] в [batch_size * seq_length, num_classes]
            label = label.view(-1)  # Из [batch_size, seq_length] в [batch_size * seq_length]

            loss = criterion(output, label)
            total_loss += loss.item()

    return total_loss / len(data_loader)


# Алгоритм обучения
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}')
