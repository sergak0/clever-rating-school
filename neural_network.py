import torch
import torch.nn as nn
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]


class LSTM_variable_input(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.3)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out

def get_seq_rating(X_test, vocab_size) :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_test = [0 for i in range(len(X_test))]
    batch_size = 100
    dataset = ReviewsDataset(X_test, y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = LSTM_variable_input(vocab_size, 100, 100).to(device)
    model.load_state_dict(torch.load("lstm_dump", map_location=torch.device('cpu')))
    model.eval()
    ans = np.array([])
    for x, y, l in dataloader:
        x = x.long().to(device)
        y = y.long().reshape(-1, 1).to(device)
        y_pred = model(x, l)
        pred = y_pred.reshape(-1).cpu().detach().numpy()
        ans = np.concatenate((ans, pred), axis = 0)
    return ans