import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, task='classification', ReVIn=False):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.task = task
        self.ReVIn = ReVIn
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.4)
        self.hidden2tag = nn.Linear(hidden_dim, output_dim)
        self.hidden = (torch.zeros(1, 256, hidden_dim), torch.zeros(1, 256, hidden_dim))

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.ReVIn:
            mean = torch.mean(x, dim=1, keepdim=True)
            std = torch.std(x, dim=1, keepdim=True)
            x = (x - mean) / (std + 0.001)
        lstm_out, _ = self.lstm(x)
        tag_space = self.hidden2tag(lstm_out)
        if self.task == 'classification':
            tag_scores = torch.softmax(tag_space, axis=-1)
        elif self.ReVIn:
            tag_scores = tag_space * (std + 0.001) + mean
        else:
            tag_scores = tag_space
        return tag_scores.transpose(1, 2)

    def predict(self, x):
        return self.forward(x) 