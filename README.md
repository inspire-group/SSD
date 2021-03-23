# SSD: A Unified Framework for Self-Supervised Outlier Detection (ICLR 2021)

Pdf: https://openreview.net/forum?id=v5gjXpmR8J

Code for our ICLR 2021 paper on outlier detection, titled SSD, without requiring class labels of in-distribution training data. We leverage recent advances in self-supervised representation learning followed by cluster based outlier detection to achieve competitive performance. This repository support both self-supervised training of networks and outlier detection evaluation of pre-trained networks. It also include code for the two proposed extensions in the paper, i.e., 1) Few-shot outlier detection and 2) Extending SSD by including class labels, when available.




## Getting started

Let's start by installing all dependencies. 

`pip install -r requirement.txt`



## Outlier detection with a pre-trained classifier

This is how we can evaluate the performance of a pre-trained ResNet50 classifier trained using SimCLR on CIFAR-10 dataset. 

`CUDA_VISIBLE_DEVICES=$gpus_ids python -u eval_ssd.py --arch resnet50 --training-mode SimCLR --dataset cifar10 --ckpt checkpoint_path --normalize --exp-name name_of_this_experiment`

`training-mode`: Choose from (`"SimCLR", "SupCon", "SupCE"`). This will choose the right network modules for the checkpoint.
`arch`: Choose from available architectures in `models.py`
`dataset`: Choose from (`"cifar10", "cifar100", "svhn", "stl"`) 
`--normalize`: If set, it will normalize input images. Use only if inputs were normalized in training too. 
`--exp-name`: Experiment name. We will log results into a text file of this name.




The steps to evaluates with $SSD_k$ are exactly same, except that now you have to also provide values for `k` and `copies` . `k` refers to how many outliers are available from each class of targeted OOD datasets while `copies` refers to number of transformed instances created per available outlier image.

`CUDA_VISIBLE_DEVICES=$gpu_id python -u eval_ssdk.py --arch resnet50 --training-mode SimCLR --dataset cifar10 --ckpt checkpoint_path --normalize --k 5 --copies 10`




## Training a classifier using self-supervised/supervised learning

We also support training a classifier using self-supervised, supervised or combination of both training methods. Here is an example script to train a ResNet50 networks on CIFAR-10 dataset using SimCLR. 

`CUDA_VISIBLE_DEVICES=$gpus_ids python -u train.py --arch resnet50 --training-mode SimCLR --dataset cifar10 --results-dir directory_to_save_checkpoint --exp-name name_of_this_experiment --warmup --normalize`

`training-mode`: Choose from (`"SimCLR", "SupCon", "SupCE"`). This will choose appropriate network modules, loss functions, and trainers.
`wamrup`: We recommend using warmup when batch-size is large, which is often that case for self-supervised methods. 

Choices for other arguments are similar to what we mentioned earlier in evaluation section.



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