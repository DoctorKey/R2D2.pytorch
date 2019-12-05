import time

import torch
import torch.nn as nn

from .utils import AverageMeterSet, accuracy

def validate(eval_loader, model, global_step, writer, LOG, print_freq, type_string=''):
    class_criterion = nn.CrossEntropyLoss().cuda()

    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()

            # compute output
            class_logit = model(input)

            class_loss = class_criterion(class_logit, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(class_logit.data, target.data, topk=(1, 5))
        minibatch_size = len(target)
        meters.update('class_loss', class_loss.item(), minibatch_size)
        meters.update('top1', prec1, minibatch_size)
        meters.update('top5', prec5, minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            LOG.info(
                'Test: [{0}/{1}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'
                .format(i, len(eval_loader), meters=meters))

    LOG.info(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
          .format(top1=meters['top1'], top5=meters['top5']))
    writer.add_scalar("{}val/prec1".format(type_string), meters['top1'].avg, global_step)
    writer.add_scalar("{}val/prec5".format(type_string), meters['top5'].avg, global_step)

    return meters['top1'].avg
