import torch
import random
import importlib
import numpy as np
from functools import partial
from collections.abc import Sequence
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from mmcv.runner import get_dist_info


def build_module(lib, args):
    assert 'type' in args
    model = getattr(lib, args.type)
    args.pop('type')

    return model(**args)


def build_dataset(cfg, data_cfg, load_ann=True, is_training=True):
    args = cfg.copy()
    data_lib = importlib.import_module('datasets')

    # init pipeline objects
    if 'pipeline' in args:
        pipeline = args.pipeline
        transforms = list()
        for item in pipeline:
            transforms.append(build_module(data_lib, item))
        args.pipeline = transforms

    # init dataset
    args['data_cfg'] = data_cfg
    args['load_ann'] = load_ann
    args['test_mode'] = not is_training
    dataset = build_module(data_lib, args)

    return dataset


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number ofhrnet_w32-36af842 workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_function(batches, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    """

    if not isinstance(batches, Sequence):
        raise TypeError(f'{batches.dtype} is not supported.')

    data = dict()
    meta_data = dict()

    for batch in batches:
        for key in batch:
            if type(batch[key]).__name__ == 'ndarray':
                if key not in data:
                    data[key] = list()
                data[key].append(torch.from_numpy(batch[key]))
            else:
                if key not in meta_data:
                    meta_data[key] = list()
                meta_data[key].append(batch[key])

    data.update(meta_data)

    return data


def collate_function_for_list(batches, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    """

    if not isinstance(batches, Sequence):
        raise TypeError(f'{batches.dtype} is not supported.')

    data = dict()
    meta_data = dict()
    for batch_list in batches:
        for batch in batch_list:
            for key in batch:
                if type(batch[key]).__name__ == 'ndarray':
                    if key not in data:
                        data[key] = list()
                    data[key].append(torch.from_numpy(batch[key]))
                else:
                    if key not in meta_data:
                        meta_data[key] = list()
                    meta_data[key].append(batch[key])

    # for key in data:
    #     data[key] = torch.stack(data[key], dim=0)

    data.update(meta_data)

    return data


def collate_function_for_episode(batches, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    """

    if not isinstance(batches, Sequence):
        raise TypeError(f'{batches.dtype} is not supported.')

    assert samples_per_gpu == 1
    return batches[0]


def build_loader(cfg, distributed, dataset, seed, num_gpus):
    rank, world_size = get_dist_info()
    samples_per_gpu = cfg.get('samples_per_gpu', 1)

    if distributed:
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=True)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = cfg.workers_per_gpu
    else:
        sampler = None
        shuffle = True
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * cfg.workers_per_gpu

    collate_fn = partial(collate_function_for_list, samples_per_gpu=samples_per_gpu)
    init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=True)

    return data_loader


def build_val_loader(dataset, batch_size, num_workers, distributed=False, fsl=True):
    rank, world_size = get_dist_info()
    if distributed:
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    if fsl:
        collate_fn = partial(collate_function_for_episode, samples_per_gpu=batch_size)
    else:
        collate_fn = partial(collate_function, samples_per_gpu=batch_size)

    val_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn)

    return val_loader
