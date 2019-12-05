import re
import os
import pickle
import sys
import random

def gen_label(datadir, labeled_nums, unlabeled_nums, output_filename):
    class_list = os.listdir(datadir)
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    f = open(output_filename, 'w')
    for i in range(len(class_list)):
        each_class = class_list[i]
        class_dir = os.path.join(datadir, each_class)
        img_file_list = os.listdir(class_dir)
        random.shuffle(img_file_list)
        labeled_num = labeled_nums[i]
        unlabeled_num = unlabeled_nums[i]
        labeled_file = img_file_list[:labeled_num]
        unlabeled_file = img_file_list[labeled_num:labeled_num + unlabeled_num]
        for filename in labeled_file:
            f.write(filename + " " + each_class + "\n")
        for filename in unlabeled_file:
            f.write(filename + " NO_LABEL\n")
    f.close()

def cifar10_4000_balanced_46000_balanced():
    datadir='images/cifar10/train+val'
    labeled_nums = [400] * 10
    unlabeled_nums = [4600] * 10
    output_filename = 'labels/cifar10/4000_balanced_labels/00.txt'
    gen_label(datadir, labeled_nums, unlabeled_nums, output_filename)

def cifar10_4000_balanced_23000_balanced():
    datadir='images/cifar10/train'
    labeled_nums = [400] * 10
    unlabeled_nums = [2300] * 10
    output_filename = 'labels/cifar10/4000_balanced_23000_balanced/00.txt'
    gen_label(datadir, labeled_nums, unlabeled_nums, output_filename)

def cifar10_4000_balanced_23000_unbalanced():
    datadir='images/cifar10/train'
    labeled_nums = [400] * 10
    unlabeled_nums = [2770, 3452, 2042, 4062, 4047, 758, 590, 2588, 2201, 490]
    output_filename = 'labels/cifar10/4000_balanced_23000_unbalanced/00.txt'
    gen_label(datadir, labeled_nums, unlabeled_nums, output_filename)

if __name__ == '__main__':
    cifar10_4000_balanced_46000_balanced()
