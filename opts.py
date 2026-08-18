import argparse
import math
import random
import os


def validate_options(opt):
    if not 0.0 <= opt.ot_weight_floor <= 1.0:
        raise ValueError('ot_weight_floor must be between 0 and 1')
    if opt.ot_candidate_k < 1:
        raise ValueError('ot_candidate_k must be at least 1')
    if not math.isfinite(opt.ot_epsilon) or opt.ot_epsilon <= 0:
        raise ValueError('ot_epsilon must be finite and positive')
    if not math.isfinite(opt.ot_rho) or opt.ot_rho <= 0:
        raise ValueError('ot_rho must be finite and positive')
    if opt.ot_max_iter < 1:
        raise ValueError('ot_max_iter must be at least 1')
    if not math.isfinite(opt.ot_tol) or opt.ot_tol <= 0:
        raise ValueError('ot_tol must be finite and positive')
    if opt.ot_block_size < 1:
        raise ValueError('ot_block_size must be at least 1')
    return opt

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for data construction, augmentation, and model training.')
    parser.add_argument('--tau', default=0.03, type=float)
    parser.add_argument('--stage', default='learning', type=str)
    parser.add_argument('--mining_method', default='mnn', choices=('mnn', 'ot'),
                        help='Pseudo-pair mining method used after MineEpoch.')

    parser.add_argument('--mining_start', default=10, type=int)
    parser.add_argument('--paired_length', default=5000, type=int)
 
    parser.add_argument('--init_txt', default='uniform',
                        help='uniform|glove only in GRU')
    parser.add_argument('--img_enhance', action='store_false',
                        help='Default is True')
    parser.add_argument('--caption_enhance', action='store_false',
                        help='Default is True')
    parser.add_argument('--use_bi_gru', action='store_false',
                        help='Default is True')
    parser.add_argument('--logger_path', default='./runs/test/checkpoint',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_path', default='./runs/test/log',
                        help='Path to save the model.')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--data_path', default='./data',
                        help='path to datasets')
    parser.add_argument('--vocab_path', default='./vocab',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--glove_cache_path', default='./vocab/vector_cache',
                        help='Path to cached GloVe vectors.')
    parser.add_argument('--glove_name', default='840B',
                        help='GloVe corpus name, e.g. 840B or 6B.')

    parser.add_argument('--MaxEpoch', default=40, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--MineEpoch', default=25, type=int)
    parser.add_argument('--memory_update_interval', default=5, type=int,
                        help='Number of mining-stage epochs between memory bank refreshes.')
    parser.add_argument('--rejected_weight_floor', default=0.5, type=float,
                        help='Minimum mining-loss weight for pseudo-pairs rejected by MNN.')
    parser.add_argument('--ot_weight_floor', default=0.0, type=float,
                        help='Minimum O2 confidence used by the mining loss.')
    parser.add_argument('--ot_candidate_k', default=32, type=int,
                        help='Top-k candidates retained in each direction for sparse O2 UOT.')
    parser.add_argument('--ot_epsilon', default=0.05, type=float,
                        help='Entropy regularization used by O2 UOT.')
    parser.add_argument('--ot_rho', default=1.0, type=float,
                        help='Marginal KL penalty used by O2 UOT.')
    parser.add_argument('--ot_max_iter', default=200, type=int,
                        help='Maximum number of O2 UOT iterations.')
    parser.add_argument('--ot_tol', default=1e-3, type=float,
                        help='O2 UOT log-scaling convergence tolerance.')
    parser.add_argument('--ot_block_size', default=1024, type=int,
                        help='Similarity block size for O2 candidate construction.')
    parser.add_argument('--ot_confidence', default='mass_concentration',
                        choices=('row_mass', 'concentration', 'mass_concentration'),
                        help='Continuous confidence stored in the O2 memory bank.')
    parser.add_argument('--UpdateEpoch', default=35, type=int)
                                      
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=.0005, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=100, type=int,
                        help='Number of steps to logger.info and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')

    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    return parser 
