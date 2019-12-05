import sys
import os
import logging

import torch

sys.path.append('.')
from src.semi_supervised import finetune_all_by_pseudo_label as main
from src.cli import parse_dict_args
from src.run_context import RunContext


LOG = logging.getLogger('runner')


def parameters():
    defaults = {
        'title': '4th finetune cifar-10',
        # Technical details
        'workers': 8,
        'checkpoint_epochs': 50,

        # Data
        'dataset': 'cifar10',
        'train_subdir': 'train+val',
        'eval_subdir': 'test',
        'exclude_unlabeled': True,

        # Data sampling
        'base_batch_size': 128,
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

        'prediction': False,

        # Optimization
        'epochs': 50,
        'lr_rampdown_epochs': 50,
        'lr_rampup': 0,
        'base_lr': 0.0025,
        'nesterov': True,
        'evaluate': False,
        'prediction': True,
        #'prediction': False,

        'n_labels': '4000_balanced_labels',
        'data_seed': 0,

        'pretrained': "results/cifar10/shakeshake/semi_4000_2_3_R2D2/2019-11-29_11:19:49/4000_balanced_labels_0/transient/checkpoint.600.ckpt",

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
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
    for run_params in parameters():
        run(**run_params)
