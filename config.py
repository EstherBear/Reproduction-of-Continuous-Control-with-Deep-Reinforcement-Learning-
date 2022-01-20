import os
import logging
import warnings
import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore")

def str2bool(input):
    if isinstance(input, bool):
        return input

    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def show_args(args):

    logger.info("==========================================")
    logger.info("==========       CONFIG      =============")
    logger.info("==========================================")

    for arg, content in args.__dict__.items():
        logger.info("{}: {}".format(arg, content))

    logger.info("==========================================")
    logger.info("===========        END        ============")
    logger.info("==========================================")

    logger.info("\n")


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch DDPG Training')

    parser.add_argument('--post', type=str, help='post of experiment')
    parser.add_argument('--mode', type=str, choices=['bn', 'tn', 'bntn', 'normalnoise'], help='mode of experiment')
    parser.add_argument('--envName', type=str, help='name of environment')
    
    parser.add_argument('--batchSize', type=int, default=64, help='mini-batch size (default: 64)')
    parser.add_argument('--warmSteps', type=int, default=2500, help='warm step (default: 2500)')
    parser.add_argument('--maxSteps', type=int, default=1000000, help='max step (default: 1000000)')
    parser.add_argument('--evaluateFreq', type=int, default=5000, help='max step (default: 5000)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    
    args = parser.parse_args()

    return args
