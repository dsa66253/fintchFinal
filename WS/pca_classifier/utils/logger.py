""" Logging """
import logging

logger = logging.getLogger(__name__)
formatter = logging.Formatter(
            '[%(levelname)s][%(asctime)s][%(module)s:%(lineno)d]: %(message)s')
streamhandler = logging.StreamHandler()
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)
logger.setLevel(logging.DEBUG)
