
import numpy as np
import torch
import argparse
import pickle
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from pca_classifier.utils.lr_schdule import BertLR
from pca_classifier.utils.trainer import Trainer
from pca_classifier.utils.configs import configs
from pca_classifier.utils.logger import logger
from torch.utils.data import TensorDataset, DataLoader
from pca_classifier.gru.model.autoencoder import AutoEncoder
import nni
from nni.utils import merge_parameter
from pca_classifier.utils.utils import make_dataloader

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='train gru encoder')
    parser.add_argument(
        '--n_hidden',
        type=int,
        default=configs.haparam['n_hidden'],
        help='input batch size for training (default: 150)')
    parser.add_argument(
        "--n_layers", 
        type=int,
        default=configs.haparam['n_layers'],
        help='number of layer (default: 2)')
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=configs.haparam['n_epochs'],
        help='hidden layer size (default: 100)')
    parser.add_argument(
        '--warmup',
        type=float,
        default=configs.haparam['warmup'],
        help='learning rate (default: 0.1)')
    parser.add_argument(
        '--decoder_drop',
        type=float,
        default=configs.haparam['decoder_drop'],
        help='decoder dropout (default: 0.5)')
    parser.add_argument(
        '--encoder_drop',
        type=float,
        default=configs.haparam['encoder_drop'],
        help='encoder dropout (default: 0.5)')
    parser.add_argument(
        '--lr',
        type=float,
        default=configs.haparam['lr'],
        help='learning rate (default: 1e-4)')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=configs.haparam['batch_size'],
        help='batch_size (default: 200)')
    parser.add_argument(
        '--train_pkl',
        default='data/train.pkl',
        type=str,
        help='where train_pkl locate'
    )
    parser.add_argument(
        '--valid_pkl',
        default='data/valid.pkl',
        type=str,
        help='where valid_pkl locate'
    )


    args, _ = parser.parse_known_args()
    return args


def main(args):
    train_dataloader = make_dataloader(args['train_pkl'])
    valid_dataloader = make_dataloader(args['valid_pkl'])


    logger.info('train data len:{}'.format(len(train_dataloader.dataset)))
    logger.info('valid data len:{}'.format(len(valid_dataloader.dataset)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_set_size = len(train_dataloader.dataset)

    model = AutoEncoder(
        n_inputs=train_dataloader.dataset[0][0].shape[1],
        n_hidden=args['n_hidden'],
        n_layers=args['n_layers'],
        device=device,
        encoder_drop=args['encoder_drop'],
        decoder_drop=args['decoder_drop'])

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_steps = int(configs.haparam['n_epochs']*train_set_size/args['batch_size'])
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args['lr'])

    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=args['warmup'],
                                     t_total=num_train_steps)
    lr_scheduler = BertLR(optimizer = optimizer,
                          learning_rate = args['lr'],
                          t_total = num_train_steps,
                          warmup=args['warmup'])
    main_trainer = Trainer(
                epoch=configs.haparam['n_epochs'],
                batch_size=args['batch_size'],
                logger=logger,
                device=device,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                lr_scheduler=lr_scheduler,
                model=model,
                num_train_steps=num_train_steps)
    main_trainer.train()

if __name__ == "__main__":
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.info(tuner_params)
        print('get_params():',get_params())
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise