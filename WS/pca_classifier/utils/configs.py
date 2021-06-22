""" Configs """
from pydantic import BaseSettings
from typing import Dict, Any
from .logger import logger
import json

class Settings(BaseSettings):
    """ Default Basic Setting Configs

    The priority goes:
        environment variable > .env file > defaults in class Settings

    """
    # The defaults settings starts here
    valid_set_size_percentage: int = 10
    test_set_size_percentage: int = 10
    no_breach_delete_percentage: int = 10
    min_seq_len: int = 10
    max_seq_len: int = 100
    haparam: Dict[str, Any] = {
        'n_epochs': 200,
        "encoder_drop": 0.6310456543266503,
        "decoder_drop": 0.6773399442974903,
        "warmup": 0.1,
        "lr": 0.0001,
        "batch_size": 200,
        "n_layers": 1,
        "n_hidden": 25

    }
    need_list: Dict[str, int] = {
        'PRICE': 10,
        'ROI': 10,
        'OPEN_ACCT_YEAR': 10,
        'BUY_COUNT': 100,
        'SELL_COUNT': 100,
        'NONTXN_COUNT': 100,
        'OPEN_PRICE': 100,
        'MAX_PRICE': 100,
        'MIN_PRICE': 100,
        'CLOSE_PRICE': 100,
        'VOLUME': 100,
        'AMONT': 100,
    }
    COMMISION_TYPE_CODE: Dict[Any, int] = {
        '0': 0,
        '1': 1,
        '2': 2,
        '9': 3,
        'A': 4
    }
    ckpt_model_path: str = 'pca_classifier/gru/checkpoint/model.ckpt'
    ckpt_model_loss: str = 'pca_classifier/gru/checkpoint/loss.txt'
    ckpt_simplemodel_path: str = 'pca_classifier/gru/checkpoint/simplemodel.ckpt'
    pca_path: str = 'pca_classifier/pca_data/pca.pkl'

configs = Settings()
logger.info('\n****************** The Followings Are The Configs ******************\n')
for k, v in configs.__dict__.items():
    logger.info('%r: %r',k,{json.dumps(v)})
logger.info('\n******************** The Aboves Are The Configs ********************\n')
