
import pickle
from pca_classifier.utils.configs import configs
from pca_classifier.gru.encoder import GruEncoder
from pca_classifier.gru.model.model import SimpleClassifier
import torch
import logging
from torch import nn

# from openTSNE import TSNE
# tsne = TSNE(
#             perplexity=30,
#             metric="euclidean",
#             n_jobs=24,
#             random_state=42,
#             verbose=True)
# trans_points = tsne.fit(model_result[0])

class PcaClassifier():
    def __init__(self, pca_path=configs.pca_path, simplemodel_path=configs.ckpt_simplemodel_path, logger=None):
        self.encoder = GruEncoder()
        self.pca = self.load_pkl(pca_path)
        # self.clf = self.load_pkl('pca_classifier/pca_data/clf.pkl')
        if logger is None:
            self.logger = self._init_logger()
        else:
            self.logger = logger
        self.device = torch.device(
                    "cuda:0" if torch.cuda.is_available()
                    else "cpu")
        self.logger.info("device:{}".format(self.device))
        self.simplemodel_path = simplemodel_path
        self.logger.info("use simplemodel in: {}".format(self.simplemodel_path))
        ckpt = torch.load(self.simplemodel_path, map_location=self.device)
        self.logger.info("start loading model")
        self.model = SimpleClassifier() 
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.sigmoid = nn.Sigmoid()
    
    def get_pca(self):
        return self.pca
    
    def _init_logger(self):
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter(
            "[%(levelname)s][%(asctime)s][%(module)s:%(lineno)d]: %(message)s")
        streamhandler = logging.StreamHandler()
        streamhandler.setFormatter(formatter)
        logger.addHandler(streamhandler)
        logger.setLevel(logging.INFO)
        return logger

    def load_pkl(self, pkl_path):
        print(f'loading {pkl_path} ...')
        with open(pkl_path, 'rb') as f:
            pkl = pickle.load(f)
        return pkl
    
    def predict(self, x, fetch_points=False):
        encoded_x = self.encoder.predict(x)
        trans_points = self.pca.transform(encoded_x)
        trans_points = torch.tensor(trans_points, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(inputs=trans_points)
            logits = self.sigmoid(logits)
            y_pred = (logits[:,1] - logits[:,0] > -0.15).int().detach().cpu().numpy()
            # y_pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
            # y_pred = logits[logits[:,1] > 0.3]
            # # y_pred = self.clf.predict(trans_points)
            if fetch_points:
                return y_pred, trans_points.detach().cpu().numpy()
            return y_pred

