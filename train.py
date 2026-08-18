# coding=utf-8
import logging
import os
import random
import time
import numpy as np
import torch 
import torch.backends.cudnn as cudnn

import shutil
import opts
import wandb

import data
from utils import save_config, cosine_similarity_matrix
from evaluation import AverageMeter, LogCollector, encode_data, evalrank, i2t, t2i
from model import SVSE
from ot_mining import mine_o2_pairs
from vocab import deserialize_vocab, deserialize_vocab_glove
import warnings

warnings.filterwarnings("ignore")


def set_random_seed(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class WandbLogger(object):
    def __init__(self):
        self.run = None

    def configure(self, opt):
        self.run = wandb.init(
            project=os.environ.get('WANDB_PROJECT', 'RCSL'),
            name=os.environ.get('WANDB_NAME'),
            dir=opt.logger_path,
            config=vars(opt),
            resume='allow' if opt.resume else None,
        )

    def log_value(self, key, value, step=None):
        wandb.log({key: value}, step=step)

    def log_values(self, values, step=None):
        wandb.log(values, step=step)

    def finish(self):
        wandb.finish()


wandb_logger = WandbLogger()


def adjust_learning_rate(model, epoch, lr_schedules):
    logger = logging.getLogger(__name__)
    """Sets the learning rate to the initial LR
       decayed by 10 every opt.lr_update epochs"""
    
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in model.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.1
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))


def init_logging(log_file_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s %(message)s')

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix='', ckpt=True,stage=''):
    logger = logging.getLogger(__name__)
    tries = 15
    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            if ckpt:
                torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + f'model_{stage}_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error

 
def UpdateMemoryBank_(data_loader, model, topK):
    memory_bank_path = model.opt.logger_path+f'/memory_bank_top{topK}.npy'

    if 'f30k' in data_loader.dataset.opt.data_name:
        bs = 1000
    else:
        bs = 400
        
    model.val_start()

    memory_bank = {
        'hard_i2t': torch.zeros((data_loader.dataset.img_length , 2*topK)).cuda(), # index sims_i2t
        'hard_t2i': torch.zeros((data_loader.dataset.length , 2*topK)).cuda() # index sims_t2i
    }

    print("compute embs")
    img_set =  data.Img_dataset(data_loader.dataset.images)
    cap_set =  data.Cap_dataset(data_loader.dataset.captions, data_loader.dataset.vocab)
    img_set_loader = torch.utils.data.DataLoader(dataset=img_set, batch_size=bs,
                                            shuffle=False,
                                            collate_fn=data.collate_fn_img,
                                            num_workers=10,
                                            drop_last=False)
    cap_set_loader = torch.utils.data.DataLoader(dataset=cap_set, batch_size=bs,
                                            shuffle=False,
                                            collate_fn=data.collate_fn_cap,
                                            num_workers=10,
                                            drop_last=False)

    img_embs = np.zeros((data_loader.dataset.img_length,1024))
    cap_embs = np.zeros((data_loader.dataset.length,1024))


    for i, data_i in enumerate(img_set_loader):
        images, image_lengths, img_ids = data_i
        with torch.no_grad():
            img_emb = model.forward_imgs(images, image_lengths)
        img_embs[img_ids] = img_emb.data.cpu()

    for i, data_i in enumerate(cap_set_loader):
        captions, caption_lengths, cap_ids = data_i
        with torch.no_grad():
            cap_emb = model.forward_caps(captions, caption_lengths)
        cap_embs[cap_ids] = cap_emb.data.cpu()

    shuffle_inx = data_loader.dataset.shuffle_inx
    i_label = torch.ones(img_embs.shape[0])
    t_label = torch.ones(cap_embs.shape[0])
    for i in range(img_embs.shape[0]):
        if shuffle_inx[i] == i:
            i_label[i] = 0

    for i in range(cap_embs.shape[0]):
        if shuffle_inx[i//5] == i//5:
            t_label[i] = 0

    i_label = i_label.cuda()
    t_label = t_label.cuda()


    print("i2t correlation")
    n_i = (img_embs.shape[0]-1) // bs +1
    n_t = (cap_embs.shape[0]-1) // bs +1
  
    for i in range(n_i):
        if i%5==0:
            print( f"image batch:  {i}")
        end =  i_label.size(0) if i == n_i-1 else (i+1)*bs
        sims = (torch.Tensor(img_embs[i * bs: end]).cuda()).mm( torch.Tensor(cap_embs).cuda().t()) * t_label * (i_label[ i * bs : end].view(-1,1))
        max = sims.topk(dim=1,k=topK)
        # print(max[0].size(),max[1].size())
        for j in range(i * bs, end):
            if i_label[j].data.item() == 1:
                memory_bank['hard_i2t'][j] = torch.cat([max[1][j-i * bs], max[0][j-i * bs]], dim=-1)
            else:
                memory_bank['hard_i2t'][j] = torch.Tensor(np.array([j*5+k for k in range(topK)]+[1 for _ in range(topK)])).cuda() 
        del sims
    print("t2i correlation")
    for i in range(n_t):
        if i%30==0:
            print( f"text batch:  {i}")
        end =  t_label.size(0) if i == n_t-1 else (i+1)*bs
        sims = (torch.Tensor(cap_embs[i * bs: end]).cuda()).mm( torch.Tensor(img_embs).cuda().t()) * i_label  * (t_label[ i * bs : end].view(-1,1))
        max = sims.topk(dim=1,k=topK)
        for j in range(i * bs, end):
            if t_label[j].data.item() == 1:
                memory_bank['hard_t2i'][j] = torch.cat([max[1][j-i * bs], max[0][j-i * bs]], dim=-1)
            else:
                memory_bank['hard_t2i'][j] = torch.Tensor(np.array([j//5 for k in range(topK)]+[1 for _ in range(topK)])).cuda() # else paired
        del sims
    memory_bank['hard_i2t'] = memory_bank['hard_i2t'].detach().cpu().numpy()
    memory_bank['hard_t2i'] = memory_bank['hard_t2i'].detach().cpu().numpy()
    
    del i_label,t_label,img_set_loader,cap_set_loader,img_set,cap_set
    torch.cuda.empty_cache()
    np.save(memory_bank_path, memory_bank)
    return memory_bank


def train(opt, train_loader, model, epoch, val_loader, best_rsum=0):
    # average meters to record the training statistics
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    num_loader_iter = len(train_loader.dataset) // train_loader.batch_size + 1
    end = time.time()
    logger.info("=======>Epoch: {0}".format(epoch))
    for i, train_data in enumerate(train_loader):
        model.train_start()
        data_time.update(time.time() - end)
        model.logger = train_logger

        # Update the model
        images, img_lengths, captions, cap_lengths, img_ids, ids, labels = train_data

        model.train_emb(images, captions, img_lengths, cap_lengths, img_ids, ids, labels, epoch=epoch)

        batch_time.update(time.time() - end)
        end = time.time()
        if model.step % opt.log_step == 0:
            logger.info( 
                'Epoch: [{0}][{1}/{2}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t{e_log}'.format(epoch, i, num_loader_iter,
                                                                                  batch_time=batch_time,
                                                                                  data_time=data_time,
                                                                                  e_log=str(model.logger)))

        # Record logs in wandb
        wandb_metrics = {
            'epoch': epoch,
            'step': i,
            'batch_time': batch_time.val,
            'data_time': data_time.val,
        }
        wandb_metrics.update({k: v.val for k, v in model.logger.meters.items()})
        wandb_logger.log_values(wandb_metrics, step=model.step)


def validate(val_loader, model, mode='dev'):
    model.val_start()
    logger.info(f"=====>Mode: {mode}")
    npts = 0
    with torch.no_grad():
        img_embs, cap_embs = encode_data(model, val_loader)
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    sims = cosine_similarity_matrix(img_embs, cap_embs)
    npts = img_embs.shape[0]
    (r1, r5, r10, medr, meanr) = i2t(npts, sims)
    logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims)
    logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                (r1i, r5i, r10i, medri, meanr))
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info('rSum is {0}'.format(currscore))
 
    # record metrics in wandb
    prefix = 't-' if mode == 'test' else ''
    wandb_logger.log_values({
        prefix + 'r1': r1,
        prefix + 'r5': r5,
        prefix + 'r10': r10,
        prefix + 'medr': medr,
        prefix + 'meanr': meanr,
        prefix + 'r1i': r1i,
        prefix + 'r5i': r5i,
        prefix + 'r10i': r10i,
        prefix + 'medri': medri,
        prefix + 'rsum': currscore,
    }, step=model.step)
        
    return currscore


def check_the_mining_quality(train_loader, model, epoch=None):
    logger = logging.getLogger(__name__)
    if model.opt.stage != 'mining':
        return

    if 'f30k' in train_loader.dataset.opt.data_name:
        bs = 1000
    else:
        bs = 400

    model.val_start()
    img_set = data.Img_dataset(train_loader.dataset.images)
    cap_set = data.Cap_dataset(train_loader.dataset.captions, train_loader.dataset.vocab)
    img_set_loader = torch.utils.data.DataLoader(dataset=img_set, batch_size=bs,
                                            shuffle=False,
                                            collate_fn=data.collate_fn_img,
                                            num_workers=train_loader.num_workers,
                                            drop_last=False)
    cap_set_loader = torch.utils.data.DataLoader(dataset=cap_set, batch_size=bs,
                                            shuffle=False,
                                            collate_fn=data.collate_fn_cap,
                                            num_workers=train_loader.num_workers,
                                            drop_last=False)

    img_embs = np.zeros((train_loader.dataset.img_length, model.opt.embed_size))
    cap_embs = np.zeros((train_loader.dataset.old_length, model.opt.embed_size))

    logger.info("compute mining quality embs")
    for _, data_i in enumerate(img_set_loader):
        images, image_lengths, img_ids = data_i
        with torch.no_grad():
            img_emb = model.forward_imgs(images, image_lengths)
        img_embs[img_ids] = img_emb.data.cpu()

    for _, data_i in enumerate(cap_set_loader):
        captions, caption_lengths, cap_ids = data_i
        with torch.no_grad():
            cap_emb = model.forward_caps(captions, caption_lengths)
        cap_embs[cap_ids] = cap_emb.data.cpu()

    shuffle_inx = train_loader.dataset.shuffle_inx
    img_unpaired = np.array([shuffle_inx[i] != i for i in range(train_loader.dataset.img_length)])
    cap_unpaired = np.array([shuffle_inx[i // train_loader.dataset.im_div] != i // train_loader.dataset.im_div
                             for i in range(train_loader.dataset.old_length)])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_embs_t = torch.Tensor(img_embs).to(device)
    cap_embs_t = torch.Tensor(cap_embs).to(device)
    best_i2t = torch.zeros(train_loader.dataset.img_length, dtype=torch.long)
    best_t2i = torch.zeros(train_loader.dataset.old_length, dtype=torch.long)

    logger.info("compute mining quality nearest neighbors")
    n_i = (img_embs_t.size(0) - 1) // bs + 1
    n_t = (cap_embs_t.size(0) - 1) // bs + 1
    for i in range(n_i):
        end = img_embs_t.size(0) if i == n_i - 1 else (i + 1) * bs
        sims = img_embs_t[i * bs:end].mm(cap_embs_t.t())
        best_i2t[i * bs:end] = sims.max(dim=1)[1].detach().cpu()
        del sims

    for i in range(n_t):
        end = cap_embs_t.size(0) if i == n_t - 1 else (i + 1) * bs
        sims = cap_embs_t[i * bs:end].mm(img_embs_t.t())
        best_t2i[i * bs:end] = sims.max(dim=1)[1].detach().cpu()
        del sims

    best_i2t = best_i2t.numpy()
    best_t2i = best_t2i.numpy()
    img_ids = np.arange(train_loader.dataset.img_length)
    cap_ids = np.arange(train_loader.dataset.old_length)

    i2t_nn_hit = img_unpaired & (best_i2t // train_loader.dataset.im_div == img_ids)
    t2i_nn_hit = cap_unpaired & (best_t2i == cap_ids // train_loader.dataset.im_div)
    i2t_mnn = img_unpaired & cap_unpaired[best_i2t] & (best_t2i[best_i2t] == img_ids)
    t2i_mnn = cap_unpaired & img_unpaired[best_t2i] & (best_i2t[best_t2i] == cap_ids)
    i2t_mnn_gt_hit = i2t_mnn & (best_i2t // train_loader.dataset.im_div == img_ids)
    t2i_mnn_gt_hit = t2i_mnn & (best_t2i == cap_ids // train_loader.dataset.im_div)

    metrics = {
        'mining_quality/i2t_nn_hit': int(i2t_nn_hit.sum()),
        'mining_quality/i2t_unpaired': int(img_unpaired.sum()),
        'mining_quality/i2t_nn_hit_rate': float(i2t_nn_hit.sum() / (img_unpaired.sum() + 1e-12)),
        'mining_quality/t2i_nn_hit': int(t2i_nn_hit.sum()),
        'mining_quality/t2i_unpaired': int(cap_unpaired.sum()),
        'mining_quality/t2i_nn_hit_rate': float(t2i_nn_hit.sum() / (cap_unpaired.sum() + 1e-12)),
        'mining_quality/i2t_mnn': int(i2t_mnn.sum()),
        'mining_quality/i2t_mnn_rate': float(i2t_mnn.sum() / (img_unpaired.sum() + 1e-12)),
        'mining_quality/i2t_mnn_gt_hit': int(i2t_mnn_gt_hit.sum()),
        'mining_quality/i2t_mnn_gt_hit_rate': float(i2t_mnn_gt_hit.sum() / (img_unpaired.sum() + 1e-12)),
        'mining_quality/i2t_mnn_precision': float(i2t_mnn_gt_hit.sum() / (i2t_mnn.sum() + 1e-12)),
        'mining_quality/t2i_mnn': int(t2i_mnn.sum()),
        'mining_quality/t2i_mnn_rate': float(t2i_mnn.sum() / (cap_unpaired.sum() + 1e-12)),
        'mining_quality/t2i_mnn_gt_hit': int(t2i_mnn_gt_hit.sum()),
        'mining_quality/t2i_mnn_gt_hit_rate': float(t2i_mnn_gt_hit.sum() / (cap_unpaired.sum() + 1e-12)),
        'mining_quality/t2i_mnn_precision': float(t2i_mnn_gt_hit.sum() / (t2i_mnn.sum() + 1e-12)),
    }
    if epoch is not None:
        metrics['epoch'] = epoch

    logger.info(
        "mining quality i2t NN GT hit: {}/{} ({:.4f}), MNN: {}/{} ({:.4f}), MNN GT hit: {} ({:.4f}), MNN precision: {:.4f}".format(
            metrics['mining_quality/i2t_nn_hit'],
            metrics['mining_quality/i2t_unpaired'],
            metrics['mining_quality/i2t_nn_hit_rate'],
            metrics['mining_quality/i2t_mnn'],
            metrics['mining_quality/i2t_unpaired'],
            metrics['mining_quality/i2t_mnn_rate'],
            metrics['mining_quality/i2t_mnn_gt_hit'],
            metrics['mining_quality/i2t_mnn_gt_hit_rate'],
            metrics['mining_quality/i2t_mnn_precision']))
    logger.info(
        "mining quality t2i NN GT hit: {}/{} ({:.4f}), MNN: {}/{} ({:.4f}), MNN GT hit: {} ({:.4f}), MNN precision: {:.4f}".format(
            metrics['mining_quality/t2i_nn_hit'],
            metrics['mining_quality/t2i_unpaired'],
            metrics['mining_quality/t2i_nn_hit_rate'],
            metrics['mining_quality/t2i_mnn'],
            metrics['mining_quality/t2i_unpaired'],
            metrics['mining_quality/t2i_mnn_rate'],
            metrics['mining_quality/t2i_mnn_gt_hit'],
            metrics['mining_quality/t2i_mnn_gt_hit_rate'],
            metrics['mining_quality/t2i_mnn_precision']))
    wandb_logger.log_values(metrics, step=model.step)

    del img_embs_t, cap_embs_t, img_set_loader, cap_set_loader, img_set, cap_set
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def com(memory_bank,th=0.5,shuffle_inx=None):
    logger = logging.getLogger(__name__)
    len_ = 0
    count_ = 0
    for i in range(memory_bank['hard_i2t'].shape[0]):
        if memory_bank['hard_i2t'][i][1] > th and shuffle_inx[i] != i:
            len_ += 1
            if memory_bank['hard_i2t'][i][0]//5 == i:
                count_ += 1
    logger.info(f"i2t hard matched:  {count_}, {len_}, {count_/(len_+1)}")
   
    len_ = 0
    count_ = 0
    for i in range(memory_bank['hard_t2i'].shape[0]):
        if memory_bank['hard_t2i'][i][1] > th and shuffle_inx[i//5] != i//5:
            len_ += 1
            if memory_bank['hard_t2i'][i][0] == i//5:
                count_ += 1
    logger.info(f"t2i hard matched:  {count_}, {len_}, {count_/(len_+1)}")


def _encode_memory_bank_embeddings(data_loader, model, batch_size):
    dataset = data_loader.dataset
    model.val_start()
    img_set = data.Img_dataset(dataset.images)
    cap_set = data.Cap_dataset(dataset.captions, dataset.vocab)
    img_set_loader = torch.utils.data.DataLoader(
        dataset=img_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data.collate_fn_img,
        num_workers=data_loader.num_workers,
        drop_last=False,
    )
    cap_set_loader = torch.utils.data.DataLoader(
        dataset=cap_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data.collate_fn_cap,
        num_workers=data_loader.num_workers,
        drop_last=False,
    )
    embedding_dim = int(model.opt.embed_size)
    img_embs = np.zeros((dataset.img_length, embedding_dim), dtype=np.float32)
    cap_embs = np.zeros((dataset.old_length, embedding_dim), dtype=np.float32)
    for images, image_lengths, img_ids in img_set_loader:
        with torch.no_grad():
            img_emb = model.forward_imgs(images, image_lengths)
        img_embs[img_ids] = img_emb.detach().cpu().numpy()
    for captions, caption_lengths, cap_ids in cap_set_loader:
        with torch.no_grad():
            cap_emb = model.forward_caps(captions, caption_lengths)
        cap_embs[cap_ids] = cap_emb.detach().cpu().numpy()
    return img_embs, cap_embs


def UpdateOTMemoryBank(data_loader, model, time_u=0):
    logger = logging.getLogger(__name__)
    opt = model.opt
    memory_bank_path = os.path.join(opt.logger_path, f'memory_bank_ot_{time_u}.npy')
    ot_config = {
        'candidate_k': int(opt.ot_candidate_k),
        'epsilon': float(opt.ot_epsilon),
        'rho': float(opt.ot_rho),
        'max_iter': int(opt.ot_max_iter),
        'tol': float(opt.ot_tol),
        'block_size': int(opt.ot_block_size),
        'confidence_mode': str(opt.ot_confidence),
    }
    if os.path.exists(memory_bank_path):
        memory_bank = np.load(memory_bank_path, allow_pickle=True).item()
        if memory_bank.get('ot_config') == ot_config:
            logger.info(f"=> loading cached O2 memory bank: {memory_bank_path}")
            return memory_bank
        logger.info("=> cached O2 memory bank configuration changed, recomputing")

    dataset = data_loader.dataset
    if dataset.im_div < 1 or dataset.old_length != dataset.img_length * dataset.im_div:
        raise ValueError('O2 requires a fixed number of captions per image group')
    embedding_batch_size = 1000 if 'f30k' in dataset.opt.data_name else 400
    logger.info("Compute embeddings for O2 memory-bank update")
    img_embs, cap_embs = _encode_memory_bank_embeddings(
        data_loader, model, embedding_batch_size
    )
    image_ids = np.arange(dataset.img_length, dtype=np.int64)
    caption_ids = np.arange(dataset.old_length, dtype=np.int64)
    caption_group_ids = caption_ids // dataset.im_div
    unpaired_image_mask = dataset.shuffle_inx != image_ids
    unpaired_caption_mask = dataset.shuffle_inx[caption_group_ids] != caption_group_ids
    unpaired_image_ids = image_ids[unpaired_image_mask]
    unpaired_caption_ids = caption_ids[unpaired_caption_mask]

    hard_i2t = np.zeros((dataset.img_length, 3), dtype=np.float32)
    hard_t2i = np.zeros((dataset.old_length, 3), dtype=np.float32)
    hard_i2t[:, 0] = image_ids * dataset.im_div
    hard_i2t[:, 1:] = 1.0
    hard_t2i[:, 0] = caption_group_ids
    hard_t2i[:, 1:] = 1.0
    diagnostics = {
        'unpaired_images': int(unpaired_image_ids.size),
        'unpaired_captions': int(unpaired_caption_ids.size),
    }

    if unpaired_image_ids.size > 0:
        device = next(model.img_enc.parameters()).device
        mined = mine_o2_pairs(
            image_embeddings=torch.from_numpy(img_embs[unpaired_image_ids]),
            caption_embeddings=torch.from_numpy(cap_embs[unpaired_caption_ids]),
            candidate_k=opt.ot_candidate_k,
            epsilon=opt.ot_epsilon,
            rho=opt.ot_rho,
            max_iter=opt.ot_max_iter,
            tol=opt.ot_tol,
            block_size=opt.ot_block_size,
            confidence_mode=opt.ot_confidence,
            device=device,
        )
        matched_caption_ids = unpaired_caption_ids[mined.i2t_indices.numpy()]
        matched_image_ids = unpaired_image_ids[mined.t2i_indices.numpy()]
        hard_i2t[unpaired_image_ids, 0] = matched_caption_ids
        hard_i2t[unpaired_image_ids, 1] = mined.i2t_scores.numpy()
        hard_i2t[unpaired_image_ids, 2] = mined.i2t_confidence.numpy()
        hard_t2i[unpaired_caption_ids, 0] = matched_image_ids
        hard_t2i[unpaired_caption_ids, 1] = mined.t2i_scores.numpy()
        hard_t2i[unpaired_caption_ids, 2] = mined.t2i_confidence.numpy()
        i2t_hits = matched_caption_ids // dataset.im_div == unpaired_image_ids
        t2i_hits = matched_image_ids == unpaired_caption_ids // dataset.im_div
        diagnostics.update(mined.diagnostics)
        if not bool(mined.diagnostics['converged']):
            logger.warning(
                "O2 UOT reached max_iter without satisfying the convergence tolerance"
            )
        diagnostics.update({
            'i2t_gt_hit_rate': float(i2t_hits.mean()),
            't2i_gt_hit_rate': float(t2i_hits.mean()),
        })

    memory_bank = {
        'hard_i2t': hard_i2t,
        'hard_t2i': hard_t2i,
        'ot_config': ot_config,
        'ot_diagnostics': diagnostics,
    }
    np.save(memory_bank_path, memory_bank)
    logger.info(f"O2 diagnostics: {diagnostics}")
    wandb_logger.log_values(
        {f'ot/{key}': value for key, value in diagnostics.items()},
        step=model.step,
    )
    del img_embs, cap_embs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return memory_bank


def UpdateMemoryBank(data_loader, model, time_u=0):
    logger = logging.getLogger(__name__)
    memory_bank_path = model.opt.logger_path+f'/memory_bank_{time_u}.npy'
    if os.path.exists(memory_bank_path):
        memory_bank = np.load(memory_bank_path, allow_pickle= True).item()
        if memory_bank['hard_i2t'].shape[1] >= 3 and memory_bank['hard_t2i'].shape[1] >= 3:
            return memory_bank
        logger.info("=> existing memory bank has no MNN accept flags, recomputing")

    if 'f30k' in data_loader.dataset.opt.data_name:
        bs = 1000
    else:
        bs = 400
        
    model.val_start()
    if time_u == 0:
        memory_bank = {
            'hard_i2t': torch.zeros((data_loader.dataset.img_length , 3)).cuda(), # index sims_i2t accepted
            'hard_t2i': torch.zeros((data_loader.dataset.old_length , 3)).cuda() # index sims_t2i accepted
        }
    else:
        memory_bank = {
            'hard_i2t': torch.Tensor(model.memory_bank['hard_i2t']).cuda(), # index sims_i2t accepted
            'hard_t2i': torch.Tensor(model.memory_bank['hard_t2i']).cuda() # index sims_t2i accepted
        }
    print("compute embs")
    img_set =  data.Img_dataset(data_loader.dataset.images)
    cap_set =  data.Cap_dataset(data_loader.dataset.captions, data_loader.dataset.vocab)
    img_set_loader = torch.utils.data.DataLoader(dataset=img_set, batch_size=bs,
                                            shuffle=False,
                                            collate_fn=data.collate_fn_img,
                                            num_workers=10,
                                            drop_last=False)
    cap_set_loader = torch.utils.data.DataLoader(dataset=cap_set, batch_size=bs,
                                            shuffle=False,
                                            collate_fn=data.collate_fn_cap,
                                            num_workers=10,
                                            drop_last=False)

    img_embs = np.zeros((data_loader.dataset.img_length,1024))
    cap_embs = np.zeros((data_loader.dataset.old_length,1024))

    for i, data_i in enumerate(img_set_loader):
        images, image_lengths, img_ids = data_i
        with torch.no_grad():
            img_emb = model.forward_imgs(images, image_lengths)
        img_embs[img_ids] = img_emb.data.cpu()

    for i, data_i in enumerate(cap_set_loader):
        captions, caption_lengths, cap_ids = data_i
        with torch.no_grad():
            cap_emb = model.forward_caps(captions, caption_lengths)
        cap_embs[cap_ids] = cap_emb.data.cpu()

    shuffle_inx = data_loader.dataset.shuffle_inx
    i_label = torch.ones(img_embs.shape[0])
    t_label = torch.ones(cap_embs.shape[0])
    for i in range(img_embs.shape[0]):
        if shuffle_inx[i] == i:
            i_label[i] = 0

    for i in range(cap_embs.shape[0]):
        if shuffle_inx[i//5] == i//5:
            t_label[i] = 0

    i_label = i_label.cuda()
    t_label = t_label.cuda()

    best_i2t = torch.zeros((data_loader.dataset.img_length, 2)).cuda()
    best_t2i = torch.zeros((data_loader.dataset.old_length, 2)).cuda()


    print("i2t correlation")
    n_i = (img_embs.shape[0]-1) // bs +1
    n_t = (cap_embs.shape[0]-1) // bs +1

    for i in range(n_i):
        if i%5==0:
            print( f"image batch:  {i}")
        end =  i_label.size(0) if i == n_i-1 else (i+1)*bs
        sims = (torch.Tensor(img_embs[i * bs: end]).cuda()).mm( torch.Tensor(cap_embs).cuda().t()) 
        # sims = (torch.Tensor(img_embs[i * bs: end]).cuda()).mm( torch.Tensor(cap_embs).cuda().t()) 
     
        max = sims.max(dim=1)
        for j in range(i * bs, end):
            best_i2t[j] = torch.Tensor(np.array([max[1][j-i * bs].data.item(), max[0][j-i * bs].data.item()])).cuda()
        del sims
    print("t2i correlation")
    for i in range(n_t):
        if i%30==0:
            print( f"text batch:  {i}")

        end =  t_label.size(0) if i == n_t-1 else (i+1)*bs
        sims = (torch.Tensor(cap_embs[i * bs: end]).cuda()).mm( torch.Tensor(img_embs).cuda().t()) 
        # sims = (torch.Tensor(cap_embs[i * bs: end]).cuda()).mm( torch.Tensor(img_embs).cuda().t())
        max = sims.max(dim=1)
        for j in range(i * bs, end):
            best_t2i[j] = torch.Tensor(np.array([max[1][j-i * bs].data.item(), max[0][j-i * bs].data.item()])).cuda()
        del sims

    accepted_i2t = 0
    mined_i2t = 0
    for j in range(data_loader.dataset.img_length):
        if i_label[j].data.item() == 1:
            mined_i2t += 1
            cap_id = int(best_i2t[j][0].data.item())
            is_mutual = t_label[cap_id].data.item() == 1 and int(best_t2i[cap_id][0].data.item()) == j
            accepted_i2t += int(is_mutual)
            memory_bank['hard_i2t'][j] = torch.Tensor(np.array([
                cap_id,
                best_i2t[j][1].data.item(),
                1 if is_mutual else 0
            ])).cuda()
        else:
            memory_bank['hard_i2t'][j] = torch.Tensor(np.array([j*5, 1, 1])).cuda()

    accepted_t2i = 0
    mined_t2i = 0
    for j in range(data_loader.dataset.old_length):
        if t_label[j].data.item() == 1:
            mined_t2i += 1
            img_id = int(best_t2i[j][0].data.item())
            is_mutual = i_label[img_id].data.item() == 1 and int(best_i2t[img_id][0].data.item()) == j
            accepted_t2i += int(is_mutual)
            memory_bank['hard_t2i'][j] = torch.Tensor(np.array([
                img_id,
                best_t2i[j][1].data.item(),
                1 if is_mutual else 0
            ])).cuda()
        else:
            memory_bank['hard_t2i'][j] = torch.Tensor(np.array([j//5, 1, 1])).cuda()

    logger.info(f"MNN i2t accepted: {accepted_i2t}/{mined_i2t}")
    logger.info(f"MNN t2i accepted: {accepted_t2i}/{mined_t2i}")
    memory_bank['hard_i2t'] = memory_bank['hard_i2t'].detach().cpu().numpy()
    memory_bank['hard_t2i'] = memory_bank['hard_t2i'].detach().cpu().numpy()
    
    del i_label,t_label,img_set_loader,cap_set_loader,img_set,cap_set
    torch.cuda.empty_cache()
    np.save(memory_bank_path, memory_bank)
    return memory_bank


if __name__ == '__main__':
    parser = opts.get_argument_parser()
    opt = opts.validate_options(parser.parse_args())
    set_random_seed(opt.seed)

    # Make dir
    if not os.path.isdir(opt.model_path):
        os.makedirs(opt.model_path)
    if not os.path.isdir(opt.logger_path):
        os.makedirs(opt.logger_path)
    # Save config
    save_config(opt, os.path.join(opt.logger_path, "config.json"))
    # logger initialization
    wandb_logger.configure(opt)
    logger = init_logging(opt.logger_path + '/log.txt')
    logger.info(f"===>PID:{os.getpid()}, GPU:[{opt.gpu}]")
    logger.info(f"Random seed: {opt.seed}; cuDNN deterministic=True, benchmark=False")
    logger.info(f"Mining method: {opt.mining_method}")
    logger.info(opt)
    # Load Vocabulary

    v_path = os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name)
    if opt.init_txt == 'glove': 
        vocab_or_tokenizer = deserialize_vocab_glove(v_path)
        word2idx = vocab_or_tokenizer.word2idx
    else:
        vocab_or_tokenizer = deserialize_vocab(v_path)
        word2idx = None
    opt.vocab_size = len(vocab_or_tokenizer)
    model = SVSE(opt,word2idx)
    logger.info(f"Mining weight floor: {model.rejected_weight_floor}")
    if not model.parallel:
        model.make_data_parallel()

    # Get data loaders
    train_loader, val_loader, test_loader = data.get_loaders(opt.data_name, vocab_or_tokenizer, opt.batch_size,
                                                             opt.workers, opt)
        
    # Load checkpoint
    start_epoch = 0
    best_rsum = 0
    lr_schedules = [opt.lr_update, 2*opt.lr_update, 3*opt.lr_update]
    
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # step is used to show logs as the continuation of another training
            model.step = checkpoint['step']
            opt.stage = checkpoint['opt'].stage
            # opt.learning_rate *= 0.1 
            model.opt = opt
            if opt.stage == 'mining':
                model.memory_bank = checkpoint['memory_bank']
                memory_bank_name = 'memory_bank_ot_0.npy' if opt.mining_method == 'ot' else 'memory_bank_0.npy'
                memory_bank_path = os.path.join(model.opt.logger_path, memory_bank_name)
                np.save(memory_bank_path,  model.memory_bank)
                
            model.reinit_optimizer()

            logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                        .format(opt.resume, start_epoch-1, best_rsum))
            validate(val_loader, model, 'dev')
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))
    #####
    # Train the Model
    logger.info("Logger path\t" + opt.logger_path)
    logger.info("Save path\t" + opt.model_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)

    # UpdateMemoryBank_(train_loader, model, topK=5)
    # exit()
    for epoch in range(start_epoch, opt.MaxEpoch):
        if epoch < opt.MineEpoch:
            logger.info(f"Learning, best_rsum:{best_rsum}") 
            model.opt.stage = 'learning' 
        else:
            model.opt.stage = 'mining' 
            logger.info(f"Mining, best_rsum:{best_rsum}") 

        train_loader.dataset.re_sort()
        train_loader.dataset.opt = model.opt 

        if epoch == opt.MineEpoch: 
            model.reinit_optimizer()  #keep
            
        if epoch >= opt.MineEpoch:
            memory_update_interval = max(1, opt.memory_update_interval)
            mining_epoch = epoch - opt.MineEpoch
            mining_round = mining_epoch // memory_update_interval
            should_update_memory = model.memory_bank is None or mining_epoch % memory_update_interval == 0
            if should_update_memory:
                logger.info(f"Start mining memory update round {mining_round}")
                if opt.mining_method == 'ot':
                    memory_bank = UpdateOTMemoryBank(train_loader, model, time_u=mining_round)
                else:
                    memory_bank = UpdateMemoryBank(train_loader, model, time_u=mining_round)
                model.memory_bank = memory_bank
            else:
                logger.info(f"Reuse mining memory bank round {mining_round}")
            train_loader.dataset.memory_bank = model.memory_bank


        adjust_learning_rate(model, epoch, lr_schedules)
        train(opt, train_loader, model, epoch, val_loader, best_rsum)
        # # evaluate on validation set
        rsum = validate(val_loader, model, 'dev')
        validate(test_loader, model, 'test')

        check_the_mining_quality(train_loader, model, epoch=epoch)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if epoch == (opt.MineEpoch-1):
            ckpt = True 
        else:
            ckpt = False
 
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'step': model.step,
            'best_rsum': best_rsum,
            'memory_bank': model.memory_bank,
            'opt': opt,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_path + '/',ckpt=ckpt,stage= model.opt.stage)
 
    logger.info(f"best_rsum:{best_rsum}")

    # Get data loader
    
    checkpoint = torch.load(opt.model_path+'/model_mining_best.pth.tar')
    opt = checkpoint['opt']
    model = SVSE(opt,word2idx)
    if not model.parallel:
        model.make_data_parallel()
     
    model.load_state_dict(checkpoint['model'])

    if 'coco' in opt.data_name:
        test_loader = data.get_test_loader('testall', opt.data_name, vocab_or_tokenizer, opt.batch_size, opt.workers,
                                           opt)
        evalrank(test_loader, model, fold5=True, logger = logger)
        evalrank(test_loader, model, fold5=False, logger =logger)
    else:
        test_loader = data.get_test_loader('test', opt.data_name, vocab_or_tokenizer, 128, opt.workers,
                                           opt)
        evalrank(test_loader, model, fold5=False, logger =logger)

    wandb_logger.finish()
