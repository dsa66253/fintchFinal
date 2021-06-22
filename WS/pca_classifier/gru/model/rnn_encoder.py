import torch
from torch import nn
class EncoderRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_layers, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = n_hidden
        self.gru = nn.GRU(n_inputs, n_hidden, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs):
        inputs = self.dropout(inputs)
        rnn_output, hidden = self.gru(inputs)
        # hidden = hidden.permute(1,0,2).reshape(inputs.shape[0], -1)
        return rnn_output, hidden