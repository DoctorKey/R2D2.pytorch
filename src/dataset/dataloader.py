import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from . import data
from ..utils import assert_exactly_one

def create_train_loader(train_dataset, args):
    train_loader = torch.utils.data.DataLoader(train_dataset,
             batch_size=args.batch_size,
             shuffle=True,
             num_workers=args.workers,
             pin_memory=True,
             drop_last=False)
    return train_loader

def create_eval_loader(eval_dataset, args):
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)
    return eval_loader

def create_subset_loaders(train_dataset, labeled_idxs, args):
    #assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])
    sampler = SubsetRandomSampler(labeled_idxs)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=False)

    subset_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=False)
    return subset_loader
    
def create_two_stream_loaders(train_dataset, unlabeled_idxs, labeled_idxs, args):
    batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_sampler=batch_sampler,
                                        num_workers=args.workers,
                                        pin_memory=True)
    return loader
