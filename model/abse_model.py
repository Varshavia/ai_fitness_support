import torch
import torch.nn as nn

class ABSEModel(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=1):
        super(ABSEModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        output = self.fc(context)
        return output, attn_weights