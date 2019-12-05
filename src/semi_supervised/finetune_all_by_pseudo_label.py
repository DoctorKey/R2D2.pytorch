# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from .. import ramps, cli
from ..eval import validate
from ..architectures import create_model
from ..run_context import RunContext
from ..dataset.datasets import get_dataset_config
from ..dataset.ImageFolder import get_semi_dataset, get_ImageFolder_dataset
from ..dataset.data import NO_LABEL
from ..dataset.dataloader import create_subset_loaders, create_eval_loader, create_train_loader
from ..utils import save_checkpoint, AverageMeterSet, accuracy, parameters_string

from ..model.DeepDecipher import DeepDecipher

LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0


def main(context):
    global global_step
    global best_prec1

    context.vis_log.add_text('hparams', cli.arg2str(args))

    checkpoint_path = context.transient_dir

    LOG.info("=> loading dataset")
    dataset_config = get_dataset_config(args.dataset, args=args)
    num_classes = dataset_config.get('num_classes')

    eval_dataset = get_ImageFolder_dataset(dataset_config.get('datadir'),
                        args.eval_subdir, 
                        dataset_config.get('eval_transformation'))

    eval_loader = create_eval_loader(eval_dataset, args=args)

    train_dataset = get_semi_dataset(dataset_config.get('datadir'),
                        args.train_subdir, 
                        dataset_config.get('train_transformation'),
                        dataset_config.get('gen_pseudo_label_transformation'),
                        args.labels)

    train_loader = create_train_loader(train_dataset, args=args)
    LOG.info("=> loaded dataset (labeled: {} unlabeled: {})".format(len(train_dataset.labeled_idxs), len(train_dataset.unlabeled_idxs))) 


    LOG.info("=> creating model '{arch}'".format(arch=args.arch))
    model = create_model(args.arch, num_classes, detach_para=False)
    LOG.info(parameters_string(model))

    D2 = DeepDecipher(len(train_dataset), num_classes, args.label_lr, 0.75)
    D2.init_pseudo_label(train_dataset)
    
    if args.pretrained:
        assert os.path.isfile(args.pretrained), "=> no pretrained found at '{}'".format(args.pretrained)
        LOG.info("=> loading pretrained from checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        LOG.info("=> loaded pretrained '{}' (epoch {})".format(args.pretrained, checkpoint['epoch']))
        if args.prediction:
            LOG.info("=> Predicting pseudo-label:")
            unlabel_loader = create_subset_loaders(train_dataset, train_dataset.unlabeled_idxs, args)
            train_dataset.gen_pseudo_label()
            pre_pseudo_label(unlabel_loader, model, D2)
            train_dataset.train()
            LOG.info('=> Predicted pseudo-label')
        else:
            LOG.info("=> loading pseudo-label")
            D2.load_state_dict(checkpoint['D2'])
            LOG.info("=> loaded pseudo-label")


    opt_param = model.parameters()
    optimizer = torch.optim.SGD(opt_param, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        D2.load_state_dict(checkpoint['D2'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {} prec1 {})".format(args.resume, checkpoint['epoch'], best_prec1))

    cudnn.benchmark = True

    if args.evaluate:
        LOG.info("Evaluating the primary model:")
        validate(eval_loader, model, global_step, context.vis_log, LOG, args.print_freq)
        return

    # update targets of unlabeled data
    train_dataset.update_target(D2.pseudo_label.data)

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        train(train_loader, model, optimizer, epoch, context.vis_log)
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            prec1 = validate(eval_loader, model, global_step, context.vis_log, LOG, args.print_freq)
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1, LOG)
    LOG.info("best_prec1 {}".format(best_prec1))

def train(train_loader, model, optimizer, epoch, writer):
    global global_step

    class_criterion = nn.CrossEntropyLoss().cuda()

    meters = AverageMeterSet()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, index) in enumerate(train_loader):
        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        input = input.cuda()
        target = target.cuda()

        class_logit = model(input)

        class_loss = class_criterion(class_logit, target)
        meters.update('class_loss', class_loss.item())

        loss = class_loss
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())

        prec1, prec5 = accuracy(class_logit.data, target.data, topk=(1, 5))
        minibatch_size = len(target)
        meters.update('top1', prec1, minibatch_size)
        meters.update('top5', prec5, minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    epoch, i, len(train_loader), meters=meters))
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar("train/class_loss", class_loss.item(), global_step)
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/prec1", prec1, global_step)
            writer.add_scalar("train/prec5", prec5, global_step)

def pre_pseudo_label(unlabel_loader, model, D2):
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, _, index) in enumerate(unlabel_loader):
        meters.update('data_time', time.time() - end)

        with torch.no_grad():
            input = input.cuda()
            # compute output
            output1 = model(input)

        D2.assign_pseudo_label(index, output1.data.cpu())

        meters.update('batch_time', time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            LOG.info(
                'Gen label: [{0}/{1}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}'.format(
                    i, len(unlabel_loader), meters=meters))
            
def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    #lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))
