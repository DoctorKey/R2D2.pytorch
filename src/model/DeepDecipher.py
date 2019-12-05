import torch
from torch.autograd import Function
from torch import nn

class DeepDecipherOp(Function):
    @staticmethod
    def forward(self, index, pseudo_label, label_lr, unlabeled_ratio):
        self.label_lr = label_lr
        self.unlabeled_ratio = unlabeled_ratio

        # batch_label does not use the same storage as pseudo_label.
        # changing batch_label will not change pseudo_label.
        batch_label = pseudo_label.index_select(0, index) #copy

        self.save_for_backward(index, pseudo_label, batch_label)

        return batch_label

    @staticmethod
    def backward(self, gradOutput):
        index, pseudo_label, batch_label = self.saved_tensors

        # Compute batch_label by gradient descent
        pseudo_label_update = self.label_lr * gradOutput.data
        batch_label.data.sub_(pseudo_label_update)

        # Only update the pseudo_label of unlabeled images
        unlabeled_num = int(index.shape[0] * self.unlabeled_ratio)
        unlabeled_index = index[:unlabeled_num]
        pseudo_label[unlabeled_index, :] = batch_label[:unlabeled_num].data.cpu()
        
        return None, None, None, None

class DeepDecipher(nn.Module):

    def __init__(self, datasize, class_num, label_lr, unlabeled_ratio=0.75):
        """
            datasize: # labeled images + # unlabeled images
            class_num: # class
            label_lr: lambda
            unlabeled_ratio: in one mini-batch, the first unlabeled_ratio images are unlabeled,
                               the rest (1 - unlabeled_ratio) images are labeled.
        """
        super(DeepDecipher, self).__init__()
        self.label_lr = label_lr
        self.unlabeled_ratio = unlabeled_ratio
        self.pseudo_label = nn.Parameter(torch.zeros(datasize, class_num))

    def forward(self, index):
        out = DeepDecipherOp.apply(index, self.pseudo_label, self.label_lr, self.unlabeled_ratio)
        return out

    def init_pseudo_label(self, dataset):
        for index in dataset.labeled_idxs:
            gt_label = dataset.imgs[index][1]
            self.pseudo_label.data[index, gt_label] = 10

    def assign_pseudo_label(self, index, pseudo_label):
        self.pseudo_label.data[index, :] = pseudo_label


if __name__ == "__main__":
    dd = DeepDecipher(10, 5, 4000, 0.75)
    index = torch.arange(4)
    gt_label = torch.randn(4, 5, requires_grad=True)
    loss = (dd(index) - gt_label).sum()
    loss.backward()
    print(dd.pseudo_label.data)
    import IPython
    IPython.embed()