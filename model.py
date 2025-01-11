import torch.nn as nn
import torch.nn.functional as F

class TicketClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, target_size):
        super(TicketClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(embed_dim, target_size)

    def forward(self, text):
        # Embedding lookup and permute to match the input for Conv1d (batch, channels, sequence)
        embedded = self.embedding(text).permute(0, 2, 1)
        conved = F.relu(self.conv(embedded))
        conved = conved.mean(dim=2)  # Global Average Pooling
        return self.fc(conved)
