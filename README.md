# R2D2.pytorch

This is the PyTorch source code for [Repetitive Reprediction Deep Decipher for Semi-Supervised Learning](https://arxiv.org/abs/1908.04345). 

## Usage

### Install the dependencies

The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

```
conda create -n pytorch python=3.6
conda activate pytorch
conda install pytorch torchvision cudatoolkit=10.0
conda install tensorboard future matplotlib tqdm
```

### Prepare CIFAR-10 Dataset

The code expects to find the data in specific directories inside the data-local directory. You can prepare the CIFAR-10 with this command:

```
./data-local/bin/prepare_cifar10.sh
cd data-local
python labels/bin/prepare_cifar10_label.py
```

Then, the images of CIFAR-10 will be saved at `data-local/images/cifar10` and the labels of CIFAR-10 will be saved at `data-local/labels/cifar10`.

### Train on CIFAR-10 with 4000 labeled images

#### Stage 1

In the first stage, we only use labeled images to train the backbone network.

```
python experiments/cifar10/shakeshake/semi_4000_1_supervised_by_gtlabel.py
```

#### Stage 2

In the second stage, we train the network and optimize pseudo-labels by R2-D2.

First, change the value of `pretrained` in `experiments/cifar10/shakeshake/semi_4000_2_3_R2D2.py`. Then, run:

```
python experiments/cifar10/shakeshake/semi_4000_2_3_R2D2.py
```

#### Stage 3

In the third stage, the backbone network is finetuned by pseudo-labels.

First, change the value of `pretrained` in `experiments/cifar10/shakeshake/semi_4000_4_finetune.py`. Then, run:

```
python experiments/cifar10/shakeshake/semi_4000_4_finetune.py
```

### Tensorboard

R2D2 will generate tensorboard log during training. You can view the tensorboard by this command:

```
tensorboard --logdir results/cifar10
```

## Citing this repository

If you find this code useful in your research, please consider citing us:

```
@inproceedings{R2D2_AAAI_2020,
	title = {Repetitive Reprediction Deep Decipher for Semi-Supervised Learning},
	author = {Wang, Guo-Hua and Wu, Jianxin},
	booktitle = {Thirty-Fourth AAAI Conference on Artificial Intelligence},
	year = {2020},
}
```


## Acknowledgement

This project is based on [Mean Teacher](https://github.com/CuriousAI/mean-teacher).
