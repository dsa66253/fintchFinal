import torch
from torch import nn
import torch.nn.functional as F
from pca_classifier.utils.configs import configs
class DecoderRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, output_size, n_layers, dropout_p=0.1, max_length=configs.max_seq_len):
        super(DecoderRNN, self).__init__()
        self.hidden_size = n_hidden
        self.gru = nn.GRU(n_inputs, n_hidden, n_layers, batch_first=True)
        self.out = nn.Linear(n_hidden, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden):
        # input = F.relu(input)
        input = self.dropout(input)
        output, hidden = self.gru(input, hidden)
        # output = self.softmax(self.out(output[0]))
        output = self.out(output)
        return output, hidden
