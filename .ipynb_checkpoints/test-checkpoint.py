import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
from torch.utils.data import Dataset

# setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IMDBDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = list(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class PyTorchModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.EmbeddingBag(vocab_size, embedding_dim, sparse=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embeddings(x)
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        linear_out = self.linear(lstm_out.squeeze(1))
        output = self.sigmoid(linear_out)
        return output

tokenizer = get_tokenizer('basic_english')
train_iter = IMDB(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# build your vocabulary 
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

embedding_dim = 32
hidden_dim = 100

model = PyTorchModel(len(vocab), embedding_dim, hidden_dim).to(device)
optimizer = Adam(model.parameters())
loss_function = nn.BCELoss()

def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(torch.tensor([float(_label == 'pos')], dtype=torch.float, device=device))
        processed_text = torch.tensor(vocab(tokenizer(_text)), dtype=torch.long, device=device)
        text_list.append(processed_text)
    return torch.cat(label_list), pad_sequence(text_list, batch_first=True)

train_dataset = IMDBDataset(list(IMDB(split='train')))
dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)

# train the model
for epoch in range(10):
    total_loss = 0
    for label, text in dataloader:
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = loss_function(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch: {epoch+1}, Loss: {total_loss/len(dataloader)}")

# save the trained model
torch.save(model.state_dict(), "trained_model.pt")
print("Trained model saved successfully.")

# Load the trained model
model.load_state_dict(torch.load("trained_model.pt"))
model.eval()

# Evaluation loop
test_iter = IMDB(split='test')
test_dataset = IMDBDataset(list(test_iter))
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)

total_correct = 0
total_samples = 0

with torch.no_grad():
    for label, text in test_dataloader:
        predicted_label = model(text)
        predicted_label = predicted_label.round()  # Round to 0 or 1
        total_correct += (predicted_label == label).sum().item()
        total_samples += len(label)

accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy * 100:.2f}%")