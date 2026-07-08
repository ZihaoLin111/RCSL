import logging
import os
import torch
from transformers import BertTokenizer
from evaluation import evalrank
import data
import argparse
from model import SVSE
from vocab import deserialize_vocab, deserialize_vocab_glove
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data',
                        help='path to datasets')
    parser.add_argument('--data_name', default=None,
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./vocab',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--model_path', default='./runs/test/checkpoint/model_mining_best.pth.tar',
                        help='Path to saved model.')       


    eval_opt = parser.parse_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    checkpoint = torch.load(eval_opt.model_path)
    opt = checkpoint['opt']
    opt.data_path = eval_opt.data_path
    if eval_opt.data_name is not None:
        opt.data_name = eval_opt.data_name
    opt.vocab_path = eval_opt.vocab_path
    opt.model_path = eval_opt.model_path
    logger.info(opt)
    logger.info(f"Load model: {opt.model_path}")
    logger.info(f"Best epoch: {checkpoint['epoch']}")
    logger.info(f"Best dev rsum: {checkpoint['best_rsum']}")
    # Load Vocabulary
    word2idx = None
    
    v_path = os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name)
    if opt.init_txt == 'glove': 
        vocab_or_tokenizer = deserialize_vocab_glove(v_path)
        word2idx = vocab_or_tokenizer.word2idx
    else:
        vocab_or_tokenizer = deserialize_vocab(v_path)
        word2idx = None
    opt.vocab_size = len(vocab_or_tokenizer)
    model = SVSE(opt,word2idx)
    if not model.parallel:
        model.make_data_parallel()
     
    model.load_state_dict(checkpoint['model'])

    # Get data loader
    if 'coco' in opt.data_name:
        test_loader = data.get_test_loader('testall', opt.data_name, vocab_or_tokenizer, opt.batch_size, opt.workers,
                                           opt)
        evalrank(test_loader, model, fold5=True, logger = logger)
        evalrank(test_loader, model, fold5=False, logger =logger)
    else:
        test_loader = data.get_test_loader('test', opt.data_name, vocab_or_tokenizer, 128, opt.workers,
                                           opt)
        evalrank(test_loader, model, fold5=False, logger =logger)
