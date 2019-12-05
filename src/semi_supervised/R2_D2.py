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
from ..dataset.data import NO_LABEL
from ..dataset.datasets import get_dataset_config
from ..dataset.ImageFolder import get_semi_dataset, get_ImageFolder_dataset
from ..dataset.dataloader import create_eval_loader, create_subset_loaders, create_two_stream_loaders
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

    labeled_idxs = train_dataset.labeled_idxs
    unlabeled_idxs = train_dataset.unlabeled_idxs

    train_loader = create_two_stream_loaders(train_dataset, unlabeled_idxs, labeled_idxs, args)
    unlabel_loader = create_subset_loaders(train_dataset, unlabeled_idxs, args)
    LOG.info("=> loaded dataset (labeled: {} unlabeled: {})".format(len(labeled_idxs), len(unlabeled_idxs))) 
    
    
    LOG.info("=> creating model '{arch}'".format(arch=args.arch))
    model = create_model(args.arch, num_classes, detach_para=False)
    LOG.info(parameters_string(model))

    D2 = DeepDecipher(len(train_dataset), num_classes, args.label_lr, 0.75)
    D2.init_pseudo_label(train_dataset)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    if args.pretrained:
        assert os.path.isfile(args.pretrained), "=> no pretrained found at '{}'".format(args.pretrained)
        LOG.info("=> loading pretrained from checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        LOG.info("=> loaded pretrained '{}' (epoch {})".format(args.pretrained, checkpoint['epoch']))
        if args.prediction:
            LOG.info("=> Predict pseudo-label:")
            train_dataset.gen_pseudo_label()
            pre_pseudo_label(unlabel_loader, model, D2)
            train_dataset.train()
            LOG.info('=> finish Predict pseudo-label')
        else:
            LOG.info("=> loading pseudo-label")
            D2.load_state_dict(checkpoint['D2'])
            LOG.info("=> loaded pseudo-label")
        

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

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        # train for one epoch
        train(train_dataset, train_loader, model, D2, optimizer, epoch, context.vis_log)
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

        #"""
        reprediction_epoch = 75 # cifar10
        if (epoch + 1) % reprediction_epoch == 0 and args.prediction:
            LOG.info("=> Predict pseudo-label:")
            train_dataset.gen_pseudo_label()
            pre_pseudo_label(unlabel_loader, model, D2)
            #check_pseudo_label_dataset(unlabel_loader)
            train_dataset.train()
            LOG.info('=> finish Predict pseudo-label')    
        #"""

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'D2': D2.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1, LOG)

def train(dataset, train_loader, model, D2, optimizer, epoch, writer):
    global global_step

    logsoftmax = nn.LogSoftmax(dim=1).cuda()
    softmax = nn.Softmax(dim=1).cuda()

    meters = AverageMeterSet()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, _, index) in enumerate(train_loader):
        # measure data loading time
        meters.update('data_time', time.time() - end)

        #unlabel_minibatch_size = target.eq(NO_LABEL).sum().item()
        #assert unlabel_minibatch_size == 0

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        input = input.cuda()

        logit1 = model(input)
 
        d2_label = D2(index)
        d2_label = d2_label.cuda()

        if args.class_loss_type == 'kl_reverse':
            class_loss = torch.mean(torch.sum(softmax(logit1) * (logsoftmax(logit1) - logsoftmax(d2_label)), dim=1))
        elif args.class_loss_type == 'kl':
            class_loss = torch.mean(torch.sum(softmax(d2_label) * (logsoftmax(d2_label) - logsoftmax(logit1)), dim=1))
        elif args.class_loss_type == 'mse':
            class_loss = torch.mean(torch.sum((softmax(d2_label) - softmax(logit1)) ** 2, dim=1))
        else:
            assert False, args.class_loss_type

        class_loss = get_current_class_weight(epoch) * class_loss
        meters.update('class_loss', class_loss.item())

        ent_loss = - torch.mean(torch.sum(softmax(logit1) * logsoftmax(logit1), dim=1))
        ent_loss = get_current_entropy_weight(epoch) * ent_loss
        meters.update('ent_loss', ent_loss.item())

        loss = class_loss + ent_loss

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())    
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(logit1.data, d2_label.data.argmax(1), topk=(1, 5))
        meters.update('top1', prec1, input.size(0))
           
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
                'Loss {meters[loss]:.4f}\t'
                'class {meters[class_loss]:.4f}\t'
                'ent {meters[ent_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                .format(epoch, i, len(train_loader), meters=meters))
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/class_loss", class_loss.item(), global_step)
            writer.add_scalar("train/ent_loss", ent_loss.item(), global_step)
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

def check_pseudo_label_dataset(loader):
    LOG.info("=> checking pseudo-label")
    for i, (input, target, index) in enumerate(loader):
        unlabel_minibatch_size = target.eq(NO_LABEL).sum().item()
        assert unlabel_minibatch_size == 0
    LOG.info("=> all data have label")


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    ngpu = 4
    reprediction_epoch = 75 # cifar10
    lr_schema = [0.02, 0.01, 0.001] # cifar10

    if epoch < reprediction_epoch:
        lr = args.lr
    elif epoch < reprediction_epoch * 2:
        lr = lr_schema[0] * ngpu
    elif epoch < reprediction_epoch * 3:
        lr = lr_schema[1] * ngpu
    else:
        lr = lr_schema[2] * ngpu

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_current_class_weight(epoch):
    return args.class_weight

def get_current_entropy_weight(epoch):
    return args.entropy

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))
