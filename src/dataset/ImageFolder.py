import os
import torch
import torchvision

from .data import NO_LABEL

class SemiSupervisedDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, train_transform=None, target_transform=None, 
                gen_pseudo_label_transformation=None, labels=None):
        super(SemiSupervisedDataset, self).__init__(root,
                                          transform=train_transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
        self.train_transform = train_transform
        self.gen_pseudo_label_transformation = gen_pseudo_label_transformation 
        self.labeled_idxs, self.unlabeled_idxs = self.relabel_dataset(labels)


    def train(self):
        self.transform = self.train_transform

    def gen_pseudo_label(self):
        self.transform = self.gen_pseudo_label_transformation

    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def relabel_dataset(self, labels):
        labeled_idxs = []
        unlabeled_idxs = []
        nouse_idxs = []
        for idx in range(len(self.imgs)):
            path, _ = self.imgs[idx]
            filename = os.path.basename(path)
            if filename in labels:
                class_name = labels[filename]
                if class_name == "NO_LABEL":
                    self.imgs[idx] = path, NO_LABEL
                    unlabeled_idxs.append(idx)
                else:
                    label_idx = self.class_to_idx[class_name]
                    self.imgs[idx] = path, label_idx
                    labeled_idxs.append(idx)
                del labels[filename]
            else:
                self.imgs[idx] = path, NO_LABEL
                nouse_idxs.append(idx)

        if len(labels) != 0:
            message = "List of unlabeled contains {} unknown files: {}, ..."
            some_missing = ', '.join(list(labels.keys())[:5])
            raise LookupError(message.format(len(labels), some_missing))

        labeled_idxs = sorted(labeled_idxs)
        if len(unlabeled_idxs) == 0:
            unlabeled_idxs = nouse_idxs

        return labeled_idxs, unlabeled_idxs

    def update_target(self, pseudo_label):
        for idx in self.unlabeled_idxs:
            label = pseudo_label[idx].argmax().item()
            path, _ = self.imgs[idx]
            self.imgs[idx] = path, label


def get_semi_dataset(datadir,
                subdir,
                train_transformation, 
                gen_pseudo_label_transformation,
                labels_file):
    traindir = os.path.join(datadir, subdir)

    if labels_file:
        with open(labels_file) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
    else:
        labels = []

    train_dataset = SemiSupervisedDataset(traindir, train_transformation,
                    gen_pseudo_label_transformation=gen_pseudo_label_transformation, labels=labels)

    return train_dataset

def get_ImageFolder_dataset(datadir,
                    subdir,  
                    transformation):
    datasetdir = os.path.join(datadir, subdir)

    dataset = torchvision.datasets.ImageFolder(datasetdir, transformation)

    return dataset