import torch
from torch import nn
import torch.nn.functional as F
from pca_classifier.utils.configs import configs

class AttnDecoderRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, output_size, n_layers, dropout_p=0.1, max_length=configs.max_seq_len):
        super(AttnDecoderRNN, self).__init__()
        self.n_hidden = n_hidden
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.n_hidden)
        self.attn = nn.Linear(self.n_hidden * 2, self.max_length)
        self.attn_combine = nn.Linear(self.n_hidden * 2, self.n_hidden)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(n_inputs, n_hidden, n_layers, batch_first=True)
        self.out = nn.Linear(self.n_hidden, self.output_size)

    def forward(self, input, hidden, encoder_outputs):

        attn_weights = F.softmax(
            self.attn(torch.cat((input[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((input[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.n_hidden)