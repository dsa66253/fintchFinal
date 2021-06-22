# from pca_classifier.gru.model.rnn_attndecoder import AttnDecoderRNN
from pca_classifier.gru.model.rnn_decoder import DecoderRNN
from pca_classifier.gru.model.rnn_encoder import EncoderRNN
from torch import nn
import torch
from pca_classifier.utils.configs import configs
import random
class AutoEncoder(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_layers, device, encoder_drop, decoder_drop):
        super(AutoEncoder, self).__init__()
        self.encoder = EncoderRNN(n_inputs, n_hidden, n_layers, dropout_p=encoder_drop)
        self.decoder = DecoderRNN(n_inputs, n_hidden, n_inputs, n_layers, dropout_p=decoder_drop)
        self.device = device
        self.teacher_forcing_ratio = 0.5
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_layers = n_layers


    def forward(self, input_tensor, target_tensor, max_length=configs.max_seq_len):

        target_length = target_tensor.size(0)

        # encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=self.device)
        # encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size)

        # reuslt_output = torch.empty(input_tensor.shape, dtype=torch.float32, device=self.device)
        # reuslt_output = torch.empty(input_tensor.shape, dtype=torch.float32, device=self.device)

        encoder_outputs, encoder_hidden = self.encoder(input_tensor )

        # decoder_input = torch.tensor([[0]], device=self.device)
        decoder_input = torch.zeros(
                    (input_tensor.shape[0],1,input_tensor.shape[2]),
                    dtype=torch.float32,
                    device=self.device)
        reuslt_output = decoder_input.permute(1,0,2)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        # use_teacher_forcing = True

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for i in range(max_length):
                # decoder_output, decoder_hidden, decoder_attention = self.decoder(
                #     decoder_input, decoder_hidden, encoder_outputs)
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                reuslt_output = torch.cat((reuslt_output, decoder_output.permute(1,0,2)), 0)
                decoder_input = target_tensor[:,i,:].unsqueeze(1)  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for i in range(max_length):
                # decoder_output, decoder_hidden, decoder_attention = self.decoder(
                #     decoder_input, decoder_hidden, encoder_outputs)
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                decoder_input = decoder_output
                reuslt_output = torch.cat((reuslt_output, decoder_output.permute(1,0,2)), 0)

        return reuslt_output[1:].permute(1,0,2)