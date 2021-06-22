""" model predictor """
import torch
import logging
from pca_classifier.utils.configs import configs
import numpy as np
from sklearn.decomposition import PCA
from pca_classifier.gru.model.rnn_encoder import EncoderRNN

class GruEncoder():
    """Predictor based on gru model (garbage)

    To use:
    >>> GE = GruEncoder()
    >>> GE.predict(x)
    m x n [0, 1, 0, 0, 0, ...]
    """
    def __init__(self,
                ckpt_path=configs.ckpt_model_path,
                logger=None):
        if logger is None:
            self.logger = self._init_logger()
        else:
            self.logger = logger
        self.device = torch.device(
                    "cuda:0" if torch.cuda.is_available()
                    else "cpu")
        self.logger.info("device:{}".format(self.device))
        self.ckpt_path = ckpt_path
        self.logger.info("use checkpoint in: {}".format(self.ckpt_path))
        ckpt = torch.load(self.ckpt_path,map_location=self.device)
        self.logger.info("start loading model")
        self.n_inputs = ckpt["n_inputs"]
        self.n_hidden = ckpt["n_hidden"]
        self.n_layers = ckpt["n_layers"]
        self.model = EncoderRNN(
                n_inputs=self.n_inputs,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers) 
        model_dict = self.model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {
                        k.replace('encoder.',''): v
                        for k, v in ckpt["state_dict"].items()
                        if k.replace('encoder.','') in model_dict
                        }
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.model.load_state_dict(pretrained_dict)
        self.model.to(self.device)


    def _init_logger(self):
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter(
            "[%(levelname)s][%(asctime)s][%(module)s:%(lineno)d]: %(message)s")
        streamhandler = logging.StreamHandler()
        streamhandler.setFormatter(formatter)
        logger.addHandler(streamhandler)
        logger.setLevel(logging.INFO)
        return logger

    def _process_data(self, x):
        x = x.float().to(self.device)
        return x


    def predict(self, x):
        x = self._process_data(x)
        self.model.eval()
        with torch.no_grad():
            _, h0 = self.model(inputs=x)
            hidden = h0.permute(1,0,2).reshape(x.shape[0], -1)
            hidden = hidden.cpu()
            return hidden
    