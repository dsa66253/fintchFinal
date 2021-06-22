""" main model """
from torch import nn

class SimpleClassifier(nn.Module):
    """SimpleClassifier Model Body
    """
    def __init__(self, n_inputs=2):
        super().__init__()
        self.l1 = nn.Linear(n_inputs, 30, bias=True)
        self.l1.weight.data.uniform_(-0.1, 0.1)
        self.l1.bias.data.zero_()
        self.l2 = nn.Linear(30, 2, bias=True)
        self.l2.weight.data.uniform_(-0.1, 0.1)
        self.l2.bias.data.zero_()
        self.acti = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        inputs = self.dropout(inputs)
        tmp_x = self.l1(inputs)
        tmp_x = self.acti(tmp_x)
        logits = self.l2(tmp_x)
        return logits