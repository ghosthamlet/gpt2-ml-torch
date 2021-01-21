
import os
import json
import time
import random
import argparse
import shutil
import logging

import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.runtime.engine import DeepSpeedEngine

import transformers
from transformers import GPT2Config, get_linear_schedule_with_warmup, \
        BertTokenizer, BertTokenizerFast

from gpt2_ml_torch.modeling_gpt2 import GPT2LMHeadModel


"""
文本生成任务微调

必须先安装 deepspeed==0.3.7

数据格式查看get_configs函数内train_data命令行参数的注释
datasets/目录下有示例数据文件
    
    
测试代码：
deepspeed --num_nodes 1 --num_gpus 1 finetune_lm.py --log_name testtest --seq_len 300 --epochs 2 --batch_size 1 --lr 5e-5 --device_ids 0 --train_data datasets/test_train.txt --valid_data datasets/test_val.txt --model_config configs/small.json --vocab models/mega-clue-tok/vocab.txt --max_data_len 1000 --no_cache
    
    
微调第一阶段：
deepspeed --num_nodes 1 --num_gpus 1 finetune_lm.py --log_name finetune_large_stage1 --seq_len 300 --epochs 3 --batch_size 1 --lr 5e-8 --device_ids 0 --train_data datasets/test_train.txt --valid_data datasets/test_val.txt --pretrained_path models/mega-clue-tok --freeze_body

微调第二阶段：
deepspeed --num_nodes 1 --num_gpus 1 finetune_lm.py --log_name finetune_large_stage2 --seq_len 300 --epochs 10 --batch_size 1 --lr 5e-8 --device_ids 0 --train_data datasets/test_train.txt --valid_data datasets/test_val.txt --pretrained_path models/finetune_large_stage1_epoch_3

"""


def get_configs():
    parser = argparse.ArgumentParser(description='GPT2')
    parser.add_argument("--lr", type=float, default=5e-5, metavar="N", help="学习率")
    parser.add_argument('--warmup_steps', default=200, type=int, required=False, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, metavar="N", help="")

    parser.add_argument('--model_config', type=str, default='configs/small.json', help="测试用的模型配置文件")
    parser.add_argument('--vocab', type=str, default='models/mega-clue-tok/vocab.txt', help="测试用的字典")
    parser.add_argument('--pretrained_path', type=str, default='', help="预训练模型目录，默认为空时，用model_config和vocab参数初始化模型从头训练，可用于快速测试代码")
    parser.add_argument('--train_data', type=str, required=True, help="训练数据文件，普通的以\n分行的txt文件，必须是utf8格式")
    parser.add_argument('--valid_data', type=str, required=True, help="验证数据文件，普通的以\n分行的txt文件，必须是utf8格式")

    parser.add_argument('--freeze_body', action='store_true', help="是否禁止微调模型主体，只微调最后一层。微调可分为两个阶段，第一阶段启用这个参数")
    parser.add_argument("--max_data_len", type=int, metavar="N", help="最大训练多少份数据，默认全部，输入较小的数字以快速测试代码")
    parser.add_argument('--log_name', type=str, required=True, help="日志名字，字母或数字，不包含特殊字符或中文")

    parser.add_argument('--no_cache', action='store_true', help="是否禁止缓存数据集的预处理操作")

    parser.add_argument('--device_ids', default='0', type=str, required=False, help="可见的GPU设备，如：0,1,2")
    parser.add_argument('--no_cuda', action='store_true', help="禁止GPU")

    parser.add_argument("--seq_len", type=int, default=300, metavar="N", help="输入长度")
    parser.add_argument("--epochs", type=int, default=10, metavar="N", help="训练轮次")
    parser.add_argument(
            "--batch_size", type=int, default=1, metavar="N", help="单个GPU上的批次大小"
    )
    parser.add_argument('--seed', type=int, default=62, help='')

    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="")
 

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    os.environ.setdefault('MASTER_PORT', '3600')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1') 
    # deepspeed launcher will setup these, so default values no effects
    os.environ.setdefault('WORLD_SIZE', str(len(args.device_ids.split(','))))
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.device_ids)

    args.deepspeed = True
    args.cpu_optimizer = True

    args.rank = int(os.getenv('RANK'))
    args.world_size = int(os.getenv("WORLD_SIZE"))

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.device = 'cuda' if args.cuda else 'cpu'

    ds_config = {
        'zero_optimization': {
            'stage': 2,
            'cpu_offload': True,
            'contiguous_gradients': True,
            # https://github.com/microsoft/DeepSpeed/issues/467
            'overlap_comm': False,
            # 'reduce_bucket_size': 50000000
            # too small will failed with large dimension size
            'reduce_bucket_size': 3000000,
            'allgather_bucket_size': 3000000
            },
        'train_batch_size': args.batch_size * args.world_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        'fp16': {
            'enabled': True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
            },
          "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing": True
          }, 
         "wall_clock_breakdown": False,
    }

    return args, ds_config


def set_random_seed(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        # Disable CuDNN
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        # XXX: for LM with almost same length input/output all the time, enable this
        torch.backends.cudnn.benchmark = False
                
 
def create_logger(log_path, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    log_fname = log_path + '/' + name + '.log'
    file_handler = logging.FileHandler(filename=log_fname)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger
     

class LMDataset(torch_data.Dataset):
    def __init__(self, args, mode, data_path, tokenizer):
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.tokenizer = tokenizer
        self.max_data_len = args.max_data_len
        self.mode = mode
        self.data_path = data_path

        cache = not args.no_cache
        cache_path = 'caches'
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        cache_file = '%s/%s-%s.json' % (cache_path, args.log_name, mode)
        
        dist.barrier() 
        if args.local_rank != 0:
              dist.barrier() 

        if cache and os.path.exists(cache_file):
            with open(cache_file) as f:
                self.features = json.loads(f.read())
        else:
            self.features = list(self._convert_to_features(self._get_exmples()))
            if len(self.features[-1]) < self.seq_len:
                self.features = self.features[:-1]
            if cache:
                with open(cache_file, 'w') as f:
                    f.write(json.dumps(self.features))
                
        if args.local_rank == 0:
            dist.barrier()

    def _get_exmples(self):
        path = self.data_path
        with open(path) as f:
            for line in tqdm(f.read().split('\n'), ascii=True):
                line = line.strip()
                if line == '':
                    continue
                yield line

    def _convert_to_features(self, examples):
        def fn():
            for i, line in enumerate(tqdm(examples, ascii=True)):
                if self.max_data_len is not None and i == self.max_data_len:
                    break

                # 加上 [SEP] token 作为分行token
                yield self.tokenizer.tokenize(line) + [self.tokenizer.sep_token]

        xs = []
        start_point = 0
        stride = self.seq_len // 2 + self.seq_len // 4
        ids = self.tokenizer.convert_tokens_to_ids([v for arr in fn() for v in arr])
        ids_len = len(ids)

        while start_point < ids_len - self.seq_len:
            x = ids[start_point: start_point + self.seq_len]
            start_point += stride
            yield x, x

        if start_point < ids_len:
            x = ids[ids_len-self.seq_len:]
            yield x, x 

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


def collate_fn(data):
    X, y = zip(*data)

    return torch.tensor(X), torch.tensor(y)


def build_model(args):
    if args.pretrained_path == '':
        config = GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config)
        tokenizer = BertTokenizerFast(args.vocab)
        # XXX: must add this, or can't tokenize special token in string to single char
        tokenizer.sanitize_special_tokens()
        info = None
    else:
        config = GPT2Config.from_pretrained(args.pretrained_path)
        model, info = GPT2LMHeadModel.from_pretrained(args.pretrained_path, 
                config=config, output_loading_info=True)
        tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_path) 
    return model, tokenizer, info
           

def get_model_tokenizer_optimizer(args):
    model, tokenizer, _ = build_model(args)

    model.half()
    model.cuda(args.local_rank)

    # XXX: all change to model parameters 
    #      (e.g. add_special_tokens) 
    #      must happen before DDP !!
    model = DDP(model, device_ids=[args.local_rank], 
            output_device=args.local_rank)

    model_obj = model.module

    if args.freeze_body:
        model_obj.transformer.requires_grad_(False)

        model_obj.transformer.wpe.requires_grad_(True)
        model_obj.transformer.emb_norm.requires_grad_(True)
        model_obj.lm_head.requires_grad_(True)
        params = [dict(params=v)
                    for v in [
                    # wte is tie with lm_head, no need run requires_grad_
                    # don't put wte in optim, params can't dup,
                    # and autodiff will calc grads two times on params in lm_head
                    # model.module.transformer.wte.parameters(),
                    model_obj.transformer.wpe.parameters(),
                    model_obj.transformer.emb_norm.parameters(),
                    model_obj.lm_head.parameters()
                ]]
    else:
        model.requires_grad_(True)
        params = model_obj.parameters()

    optimizer = DeepSpeedCPUAdam(params, lr=args.lr, weight_decay=0.01)

    return model, tokenizer, optimizer


def get_data_loader(args, tokenizer):
    def fn(mode, data_path):
        dataset = LMDataset(args, mode, data_path, tokenizer)
        sampler = torch_data.distributed.DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True,
        )
        data_loader = torch_data.DataLoader(
                dataset=dataset, 
                batch_size=int(args.batch_size / args.gradient_accumulation_steps),
                shuffle=False, 
                collate_fn=collate_fn,
                pin_memory=True,
                sampler=sampler) 

        return data_loader, sampler

    train = fn('train', args.train_data)
    valid = fn('valid', args.valid_data)
    return (*train, *valid)
 

def get_scheduler(args, optimizer, data_loader):
    total_steps = int(len(data_loader.dataset) * args.epochs / args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
            num_warmup_steps=args.warmup_steps, num_training_steps=total_steps) 

    return scheduler


def train_epoch(args, logger, model, data_loader, optimizer, 
        scheduler, epoch=0, log_interval=10):
    model.train()

    loss_acc = 0
    oom_time = 0
    samples_so_far = 0
    batch_len = len(data_loader)
    data_len = len(data_loader.dataset)

    for batch_idx, (input_t, target_t) in enumerate(data_loader):
        input_t = input_t.cuda(args.local_rank, non_blocking=True)
        target_t = target_t.cuda(args.local_rank, non_blocking=True)
        
        loss = torch.tensor(0)
        try:
            loss, *_ = model(input_t, labels=target_t)
            loss_acc += loss.item()

            model.backward(loss)
            model.step()
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                oom_time += 1
                logger.info("WARNING: ran out of memory, times: {}".format(oom_time))
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

        if args.local_rank == 0:
            samples_so_far += len(input_t)
            if batch_idx % log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch + 1,
                        samples_so_far,
                        data_len,
                        100 * samples_so_far / data_len,
                        loss.item(),
                    )
                )

            if batch_idx > 0 and batch_idx % 2000 == 0:
                save_model(args, logger, model, epoch, batch_idx)

    return loss_acc / batch_len


def eval_epoch(args, logger, model, data_loader,
        epoch=0, log_interval=10):
    model.eval()

    loss_acc = 0
    samples_so_far = 0
    batch_len = len(data_loader)
    data_len = len(data_loader.dataset)

    with torch.no_grad():
        for batch_idx, (input_t, target_t) in enumerate(data_loader):
            input_t = input_t.cuda(args.local_rank, non_blocking=True)
            target_t = target_t.cuda(args.local_rank, non_blocking=True)

            loss, *_ = model(input_t, labels=target_t) 
            loss = average_distributed_scalar(args, loss.item())
            loss_acc += loss

            if args.local_rank == 0:
                samples_so_far += len(input_t)
                if batch_idx % log_interval == 0:
                    logger.info(
                        "Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch + 1,
                            samples_so_far,
                            data_len,
                            100 * samples_so_far / data_len,
                            loss,
                        )
                    )
 
    return loss_acc / batch_len


def average_distributed_scalar(args, scalar):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / dist.get_world_size()
    dist.all_reduce(scalar_t, op=dist.ReduceOp.SUM)
    return scalar_t.item()


def train(args, ds_config):
    dist_init(args)
 
    set_random_seed(args)

    logger = create_logger('logs/', args.log_name)

    model, tokenizer, optimizer = get_model_tokenizer_optimizer(args)
    data_loader, sampler, valid_data_loader, valid_sampler = get_data_loader(args, tokenizer)

    scheduler = get_scheduler(args, optimizer, data_loader)

    model, optimizer, _, scheduler = deepspeed.initialize(
        model=model, 
        optimizer=optimizer, 
        lr_scheduler=scheduler,
        args=args, 
        config_params=ds_config
    ) 

    for epoch in range(args.epochs):
        start = time.time()
        sampler.set_epoch(epoch)

        if args.local_rank == 0:
            logger.info("\nEpoch %s" % (epoch + 1))

        train_loss = train_epoch(args, logger, model, data_loader, 
                optimizer, scheduler, epoch)
        valid_loss = eval_epoch(args, logger, model, valid_data_loader,
                epoch)

        if args.local_rank == 0:
            end = time.time()
            logger.info("Epoch took: {:.3f}s, train loss: {:.6f}, valid loss: {:.6f}".format(
                end - start, train_loss, valid_loss))

            save_model(args, logger, model, epoch, None)

    dist_cleanup()


def dist_init(args):
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR')
    master_port = os.getenv('MASTER_PORT')
    init_method += master_ip + ':' + master_port
    dist.init_process_group(
        backend='nccl',
        world_size=args.world_size, 
        rank=args.local_rank,
        init_method=init_method)


def dist_cleanup():
    dist.destroy_process_group()


def save_model(args, logger, model, epoch, batch):
    parent = 'models'
    if not os.path.exists(parent):
        os.mkdir(parent)

    path = '{}/{}_epoch_{}'.format(parent, args.log_name, epoch+1)
    if not os.path.exists(path):
        os.mkdir(path)
    if batch is None:
        model_filename = path + '/pytorch_model.bin'
    else:
        model_filename = path + '/pytorch_model_{}.bin'.format(batch + 1)

    model_obj = model
    while isinstance(model_obj, (DeepSpeedEngine, DDP)):
        model_obj = model_obj.module

    if args.pretrained_path != '':
        config_file = args.pretrained_path + '/config.json'
        vocab_file = args.pretrained_path + '/vocab.txt'
    else:
        config_file = args.model_config
        vocab_file = args.vocab

    torch.save(model_obj.state_dict(), model_filename)

    if not os.path.exists(path + '/config.json') \
            or not os.path.samefile(config_file, path + '/config.json'):
        shutil.copy(config_file, path + '/config.json')
    if not os.path.exists(path + '/vocab.txt') \
            or not os.path.samefile(vocab_file, path + '/vocab.txt'):
        shutil.copy(vocab_file, path)

    torch.save(args, path + '/model_training_args.bin')


if __name__ == "__main__":
    args, ds_config = get_configs()

    log_path = 'logs'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logger = create_logger('logs/', args.log_name)

    def log_configs():
        logger.info(
            json.dumps({k: v for k, v in os.environ.items() if k not in ['LS_COLORS']})
        )
        logger.info(json.dumps(args.__dict__, indent=True))
        model_config_file = args.model_config if args.pretrained_path == '' \
                else args.pretrained_path + '/config.json'
        with open(model_config_file) as f:
            model_config = json.loads(f.read())
        logger.info(json.dumps(model_config, indent=True))
        logger.info(json.dumps(ds_config, indent=True))

    log_configs()

    train(args, ds_config)

