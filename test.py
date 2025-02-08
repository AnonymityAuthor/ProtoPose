import os
import torch
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.multiprocessing.queue import Queue
from torch.nn.parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from mmcv import Config
from models import build_trainer
from datasets import build_dataset, build_val_loader
from train import evaluating
from utils import get_logger, get_rank, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='Path of config file')
    parser.add_argument('mode_path', type=str,
                        help='Path of checkpoint file')
    parser.add_argument('--num_shots', type=int, default=1,
                        help='number of support images')
    parser.add_argument('--set', type=str, default='test',
                        help='test or val')
    parser.add_argument('--ft_steps', type=int, default=40,
                        help='number of finetune steps')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Gpu id in single gpu mode')
    parser.add_argument('--out_dir', type=str, default='./output',
                        help='saving directory')
    parser.add_argument('--distributed', type=bool, default=False,
                        help='if True, use Distributed Data Parallel')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus for training')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes for distributed training')
    parser.add_argument('--nr', type=int, default=0,
                        help='ranking within the nodes for distributed training')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='number of nodes for distributed training')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1',
                        help='master address used to set up distributed training')
    parser.add_argument('--master_port', type=str, default='1234',
                        help='master port used to set up distributed training')

    return parser.parse_args()


def main_worker(gpu, cfg, args, results_queue=None):
    set_random_seed(0)
    # init environment
    if args.distributed:
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        rank = args.nr * args.num_gpus + gpu
        torch.cuda.set_device(rank % args.num_gpus)
        dist.init_process_group(
            backend=args.backend,
            init_method='env://',
            world_size=args.num_gpus * args.num_nodes,
            rank=rank
        )

    # init logger
    logger = get_logger(__name__.split('.')[0])

    # init dataset
    logger.info('init dataset...')
    if args.set == 'test':
        cfg.set_cfg.test.num_shots = args.num_shots
        dataset = build_dataset(cfg.set_cfg.test, cfg.data_cfg, is_training=False)
    elif args.set == 'val':
        cfg.set_cfg.val.num_shots = args.num_shots
        dataset = build_dataset(cfg.set_cfg.val, cfg.data_cfg, is_training=False)
    else:
        raise ValueError('Unknown set name: {}'.format(args.set))

    val_loader = build_val_loader(dataset, 1, cfg.eval_cfg.workers, args.distributed)

    # init model
    cfg.trainer.num_shots = args.num_shots
    cfg.trainer.ft_steps = args.ft_steps
    model = build_trainer(cfg.trainer)

    # put model on gpus
    if args.distributed:
        model = DistributedDataParallel(model.cuda(gpu), device_ids=[gpu], find_unused_parameters=True)
    else:
        model = DataParallel(model.cuda(gpu[0]), device_ids=gpu)

    logger.info('init model...')
    checkpoint = torch.load(args.mode_path, map_location='cpu')
    model.module.pose_model.load_state_dict(checkpoint['model'], strict=True)
    model.module.prompt_embeds_dict.load_state_dict(checkpoint['prompt_embeds_dict'], strict=True)

    # evaluate
    logger.info('Fwe-shot Evaluating....')
    rank = get_rank()
    results = evaluating(model, val_loader, cfg.solver.ft_optimizer, cfg.eval_cfg, rank, args.distributed)
    if args.distributed:
        if rank != 0:
            results_queue.put(results, block=True)
        else:
            for _ in range(args.num_gpus - 1):
                results += results_queue.get(block=True)

    if rank == 0:
        score, info_str = val_loader.dataset.evaluate(results, args.out_dir, cfg.eval_cfg.metric)
        logger.info('Validation Result of {}: {}'.format(val_loader.dataset.dataset_name, info_str))

    if args.distributed:
        dist.barrier()


if __name__ == '__main__':
    # load args and configs
    args = parse_args()
    cfg = Config.fromfile(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    # launch main worker
    if args.distributed:
        results_queue = Queue(maxsize=args.num_gpus - 1, ctx=mp.get_context(method='spawn'))
        mp.spawn(main_worker, nprocs=args.num_gpus, args=(cfg, args, results_queue))
    else:
        assert args.num_gpus == 1
        main_worker([i for i in range(args.num_gpus)], cfg, args)
