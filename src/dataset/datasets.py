
import torch
import torchvision
import torchvision.transforms as transforms

from . import data
from ..utils import export

@export
def cifar10(args=None):
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])

    train_trans_list = []
    if args.random_translate:
        train_trans_list.append(data.RandomTranslateWithReflect(4))
    if args.random_flip:
        train_trans_list.append(transforms.RandomHorizontalFlip())
    train_trans_list.append(transforms.ToTensor())   
    train_trans_list.append(transforms.Normalize(**channel_stats))
    if args.cutout:
        train_trans_list.append(data.Cutout(n_holes=1, length=16))

    if args.trans_form_twice:
        train_transformation = data.TransformTwice(transforms.Compose(train_trans_list))
    else:
        train_transformation = transforms.Compose(train_trans_list)

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats),
    ])
    gen_pseudo_label_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats),
    ])    

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'gen_pseudo_label_transformation': gen_pseudo_label_transformation,
        'datadir': 'data-local/images/cifar10',
        'num_classes': 10
    }


def get_dataset_config(dataset_name, args=None):
    dataset_factory = globals()[dataset_name]
    params = dict(args=args)
    dataset_config = dataset_factory(**params)
    return dataset_config
    
