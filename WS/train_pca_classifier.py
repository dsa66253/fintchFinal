from pca_classifier.gru.encoder import GruEncoder
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
from sklearn.decomposition import PCA
from pca_classifier.utils.utils import (make_dataloader, write_file)
from sklearn.ensemble import IsolationForest
from pca_classifier.utils.configs import configs
from pca_classifier.gru.model.model import SimpleClassifier
import numpy as np
import torch
import pickle
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from pca_classifier.utils.lr_schdule import BertLR
from pca_classifier.utils.simple_trainer import Trainer
from pca_classifier.utils.configs import configs
from pca_classifier.utils.logger import logger
from torch.utils.data import TensorDataset, DataLoader
from pca_classifier.utils.utils import (make_dataloader, make_dataloader_from_np)
def get_gru_encoded_data(GE, data_path, shuffle=False):
    logger.info(f'get_gru_encoded_data: {data_path}')
    dataloader = make_dataloader(data_path, shuffle=shuffle)
    hidden_array = None
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        hidden = GE.predict(x_batch)
        if hidden_array is None:
            hidden_array = hidden
        else:
            hidden_array = torch.cat((hidden_array, hidden), 0)
    return hidden_array, dataloader.dataset[:][1].numpy()

def main():
    GE = GruEncoder(logger=logger)
    train_hidden_array, y_train = get_gru_encoded_data(GE, 'data/train.pkl', shuffle=False)
    valid_hidden_array, y_valid = get_gru_encoded_data(GE, 'data/valid.pkl', shuffle=False)

    logger.info('fit pca ...')
    pca = PCA(n_components=2).fit(train_hidden_array)
    write_file(configs.pca_path, pca)

    logger.info('use pca transform data ...')
    train_trans_points = pca.transform(train_hidden_array)
    valid_trans_points = pca.transform(valid_hidden_array)

    logger.info('make data loader ...')
    train_loader = make_dataloader_from_np(train_trans_points, y_train)
    valid_loader = make_dataloader_from_np(valid_trans_points, y_valid)


    # clf = IsolationForest(random_state=0, behaviour="new").fit(train_trans_points)
    # write_file('pca_classifier/pca_data/clf.pkl', clf)

    lr = 1e-5
    n_epochs = 100
    batch_size = 200
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_set_size = len(train_loader.dataset)
    model = SimpleClassifier()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_steps = int(n_epochs*train_set_size/batch_size)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=lr)

    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=0.1,
                                     t_total=num_train_steps)
    lr_scheduler = BertLR(optimizer = optimizer,
                          learning_rate = lr,
                          t_total = num_train_steps,
                          warmup=0.1)
    main_trainer = Trainer(
                epoch=n_epochs,
                batch_size=batch_size,
                logger=logger,
                device=device,
                train_dataloader=train_loader,
                valid_dataloader=valid_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                lr_scheduler=lr_scheduler,
                model=model,
                num_train_steps=num_train_steps)
    main_trainer.train()

if __name__ == '__main__':
    main()
# %%
