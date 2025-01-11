import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import TicketClassifier
from sklearn.model_selection import train_test_split
import json

# Load data
with open('data/words.json', 'r') as f:
    words = json.load(f)
with open('data/text.json', 'r') as f:
    text = json.load(f)
labels = np.load('data/labels.npy')

# Process data (same as in your original code)
word2idx = {o: i for i, o in enumerate(words)}
text = [[word2idx.get(word, 0) for word in sentence] for sentence in text]

def pad_input(sentences, seq_len=50):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

text = pad_input(text, 50)

# Split data
train_text, test_text, train_label, test_label = train_test_split(text, labels, test_size=0.2, random_state=42)
train_data = TensorDataset(torch.from_numpy(train_text), torch.from_numpy(train_label).long())
test_data = TensorDataset(torch.from_numpy(test_text), torch.from_numpy(test_label).long())

train_loader = DataLoader(train_data, shuffle=True, batch_size=400)
test_loader = DataLoader(test_data, shuffle=False, batch_size=400)

# Model
vocab_size = len(word2idx) + 1
embedding_dim = 64
target_size = len(np.unique(labels))
model = TicketClassifier(vocab_size, embedding_dim, target_size)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

epochs = 3
for epoch in range(epochs):
    model.train()
    running_loss, num_processed = 0, 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_processed += len(inputs)
    print(f"Epoch {epoch+1}, Loss: {running_loss/num_processed}")

# Save model
torch.save(model.state_dict(), 'saved_models/model.pth')
