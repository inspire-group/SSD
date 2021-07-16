# SSD: A Unified Framework for Self-Supervised Outlier Detection [ICLR 2021]

Pdf: https://openreview.net/forum?id=v5gjXpmR8J

Code for our ICLR 2021 paper on outlier detection, titled SSD, without requiring class labels of in-distribution training data. We leverage recent advances in self-supervised representation learning followed by the cluster-based outlier detection to achieve competitive performance. This repository support both self-supervised training of networks and outlier detection evaluation of pre-trained networks. It also includes code for the two proposed extensions in the paper, i.e., 1) Few-shot outlier detection and 2) Extending SSD by including class labels, when available.




## Getting started

Let's start by installing all dependencies. 

`pip install -r requirement.txt`



## Outlier detection with a pre-trained classifier

This is how we can evaluate the performance of a pre-trained ResNet50 classifier trained using SimCLR on the CIFAR-10 dataset. 

`CUDA_VISIBLE_DEVICES=$gpus_ids python -u eval_ssd.py --arch resnet50 --training-mode SimCLR --dataset cifar10 --ckpt checkpoint_path --normalize --exp-name name_of_this_experiment`

* `training-mode`: Choose from (`"SimCLR", "SupCon", "SupCE"`). This will choose the right network modules for the checkpoint.
* `arch`: Choose from available architectures in `models.py`
* `dataset`: Choose from (`"cifar10", "cifar100", "svhn", "stl"`) 
* `--normalize`: If set, it will normalize input images. Use only if inputs were normalized in training too. 
* `--exp-name`: Experiment name. We will log results into a text file of this name.


The steps to evaluate with $SSD_k$ are exactly the same, except that now you have to also provide values for `k` and `copies` . `k` refers to how many outliers are available from each class of targeted OOD datasets while `copies` refers to the number of transformed instances created per available outlier image.

`CUDA_VISIBLE_DEVICES=$gpu_id python -u eval_ssdk.py --arch resnet50 --training-mode SimCLR --dataset cifar10 --ckpt checkpoint_path --normalize --k 5 --copies 10`



## Training a classifier using self-supervised/supervised learning

We also support training a classifier using self-supervised, supervised or a combination of both training methods. Here is an example script to train a ResNet50 network on the CIFAR-10 dataset using SimCLR. 

`CUDA_VISIBLE_DEVICES=$gpus_ids python -u train.py --arch resnet50 --training-mode SimCLR --dataset cifar10 --results-dir directory_to_save_checkpoint --exp-name name_of_this_experiment --warmup --normalize`

* `--training-mode`: Choose from (`"SimCLR", "SupCon", "SupCE"`). This will choose appropriate network modules, loss functions, and trainers.
* `--warmup`: We recommend using warmup when batch-size is large, which is often the case for self-supervised methods. 

Choices for other arguments are similar to what we mentioned earlier in the evaluation section.


## Pre-trained models
Here is the link to pre-trained models on cifar-10 dataset: https://drive.google.com/drive/folders/1Nx5tYGecvwagVz7_y8Z3FPk-ZtYttM4k?usp=sharing 

These models aren't exactly identical to ones in the paper but they give fairly similar results. Here is my attempt at doing OOD detection with the SimCLR trained models on CIFAR10. 
`CUDA_VISIBLE_DEVICES=0 python -u eval_ssd.py --arch resnet50 --training-mode SimCLR --dataset cifar10 --normalize --ckpt ./cifar10/base1/SimCLR_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_900_temp_0.5_trial_0_ps__cosine_warm/last.pth`

It gives the following results, which is fairly similar to ones in the paper. Most likely mistake, which gives suboptimal results, is to miss --normalize (when the model is trained with it). 
```
In-data = cifar10, OOD = cifar100, Clusters = 1, FPR95 = 0.5078, AUROC = 0.9063240349999999, AUPR = 0.8919609510086947
In-data = cifar10, OOD = svhn, Clusters = 1, FPR95 = 0.020666871542716656, AUROC = 0.9962383988936693, AUPR = 0.9985624119973668
In-data = cifar10, OOD = texture, Clusters = 1, FPR95 = 0.14645390070921985, AUROC = 0.9761002304964539, AUPR = 0.9556574287665671
In-data = cifar10, OOD = blobs, Clusters = 1, FPR95 = 0.0467, AUROC = 0.9879078399999999, AUPR = 0.9843056376364349
```
## Reference

If you find this work helpful, consider citing it. 

```
@inproceedings{sehwag2021ssd,
  title={SSD:  A Unified Framework for Self-Supervised Outlier Detection},
  author={Vikash Sehwag and Mung Chiang and Prateek Mittal},
 booktitle={International Conference on Learning Representations},
 year={2021},
 url={https://openreview.net/forum?id=v5gjXpmR8J}
}
```
