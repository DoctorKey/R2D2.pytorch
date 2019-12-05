import os
import sys
import logging

import torch

sys.path.append(".")
from src.semi_supervised import supervised_by_subset as main
from src.cli import parse_dict_args
from src.run_context import RunContext

LOG = logging.getLogger('runner')


def parameters():
    defaults = {
        'title': '1 4000-only labeled cifar-10',       
        # Technical details
        'workers': 2,
        'checkpoint_epochs': 50,

        # Data
        'dataset': 'cifar10',
        'train_subdir': 'train+val',
        'eval_subdir': 'test',
        'exclude_unlabeled': True,

        # Data sampling
        'base_batch_size': 40,
        'eval_batch_size': 128,

        # Architecture
        'arch': 'cifar_shakeshake26',

        # train data transform
        'random_translate': True,
        'random_flip': True,
        'cutout': True,
        'trans_form_twice': False,

        # Costs
        'weight_decay': 0.0002,

        # Optimization
        'epochs': 300,
        'lr_rampup': 0,
        'base_lr': 0.05,
        'lr_rampdown_epochs': 350,
        'nesterov': True,

        'n_labels': '4000_balanced_labels',
        'data_seed': 0,
    }

    yield {
        **defaults,        
    }


def run(title, base_batch_size, base_lr, n_labels, data_seed, **kwargs):
    LOG.info('run title: %s, data seed: %d', title, data_seed)

    ngpu = torch.cuda.device_count()
    assert ngpu > 0, "Expecting at least one GPU, found none."

    adapted_args = {
        'batch_size': base_batch_size * ngpu,
        'lr': base_lr * ngpu,
        'labels': 'data-local/labels/cifar10/{}/{:02d}.txt'.format(n_labels, data_seed),
    }
    context = RunContext(__file__, "{}_{}".format(n_labels, data_seed))
    main.args = parse_dict_args(**adapted_args, **kwargs)
    main.main(context)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    for run_params in parameters():
        run(**run_params)
