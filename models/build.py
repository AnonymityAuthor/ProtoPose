import importlib
import copy
from typing import Dict
from .trainer import Trainer


def build_model(args):
    model_type = args['type']
    args.pop('type')
    module_lib = importlib.import_module('models.pose')
    for item in args:
        sub_args = args[item]
        if isinstance(sub_args, dict) and 'type' in sub_args:
            module = getattr(module_lib, sub_args['type'])
            sub_args.pop('type')
            args[item] = module(**sub_args)

    model_m = getattr(module_lib, model_type)

    return model_m(**args)


def build_criterion(args):
    module_lib = importlib.import_module('models.loss')
    for item in args:
        sub_args = args[item]
        if isinstance(sub_args, dict) and 'type' in sub_args:
            module = getattr(module_lib, sub_args['type'])
            sub_args.pop('type')
            args[item] = module(**sub_args)

    return args


def build_trainer(args):
    args.pose_model = build_model(args.pose_model)
    args.criterion = build_criterion(args.criterion)

    return Trainer(**args)


def build_optimizer_constructor(cfg: Dict):
    module_lib = importlib.import_module('models')
    module = getattr(module_lib, cfg['type'])
    cfg.pop('type')

    return module(**cfg)


def build_optimizer(model, cfg: Dict):
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))

    if hasattr(model, 'module'):
        model = model.module
    optimizer = optim_constructor(model)

    return optimizer
