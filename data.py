import os
import numpy as np
from skimage.filters import gaussian as gblur
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler


# ref: https://github.com/HobbitLong/SupContrast
class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def cifar10(
    data_dir, batch_size=32, mode="base", normalize=True, norm_layer=None, size=32
):
    """
    mode: org | base | ssl
    """
    transform_train = [
        transforms.Resize(size),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    transform_test = [transforms.Resize(size), transforms.ToTensor()]

    if mode == "org":
        None
    elif mode == "base":
        transform_train = [transforms.Resize(size), transforms.ToTensor()]
    elif mode == "ssl":
        transform_train = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
    else:
        raise ValueError(f"{mode} mode not supported")

    if norm_layer is None:
        norm_layer = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201]
        )
    
    if normalize:
        transform_train.append(norm_layer)
        transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if mode == "ssl":
        transform_train = TwoCropTransform(transform_train)

    trainset = datasets.CIFAR10(
        root=os.path.join(data_dir, "cifar10"),
        train=True,
        download=True,
        transform=transform_train,
    )
    testset = datasets.CIFAR10(
        root=os.path.join(data_dir, "cifar10"),
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader, norm_layer


def cifar100(
    data_dir, batch_size, mode="base", normalize=True, norm_layer=None, size=32
):
    """
    mode: org | base | ssl
    """
    transform_train = [
        transforms.Resize(size),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    transform_test = [transforms.Resize(size), transforms.ToTensor()]

    if mode == "org":
        None
    elif mode == "base":
        transform_train = [transforms.Resize(size), transforms.ToTensor()]
    elif mode == "ssl":
        transform_train = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
    else:
        raise ValueError(f"{mode} mode not supported")

    if norm_layer is None:
        norm_layer = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        )

    if normalize:
        transform_train.append(norm_layer)
        transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if mode == "ssl":
        transform_train = TwoCropTransform(transform_train)

    trainset = datasets.CIFAR100(
        root=os.path.join(data_dir, "cifar100"),
        train=True,
        download=True,
        transform=transform_train,
    )
    testset = datasets.CIFAR100(
        root=os.path.join(data_dir, "cifar100"),
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader, norm_layer


def svhn(data_dir, batch_size, mode="base", normalize=True, norm_layer=None, size=32):
    """
    mode: org | base | ssl
    """
    transform_train = [
        transforms.Resize(size),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    transform_test = [transforms.Resize(size), transforms.ToTensor()]

    if mode == "org":
        None
    elif mode == "base":
        transform_train = [transforms.Resize(size), transforms.ToTensor()]
    elif mode == "ssl":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )
    else:
        raise ValueError(f"{mode} mode not supported")

    if norm_layer is None:
        norm_layer = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    if normalize:
        transform_train.append(norm_layer)
        transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if mode == "ssl":
        transform_train = TwoCropTransform(transform_train)

    trainset = datasets.SVHN(
        root=os.path.join(data_dir, "svhn"),
        split="train",
        download=True,
        transform=transform_train,
    )
    testset = datasets.SVHN(
        root=os.path.join(data_dir, "svhn"),
        split="test",
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader, norm_layer


def stl(data_dir, batch_size, mode="base", normalize=True, norm_layer=None, size=96):
    """
    mode: org | base | ssl
    """
    transform_train = [
        transforms.Resize(size),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    transform_test = [transforms.Resize(size), transforms.ToTensor()]

    if mode == "org":
        None
    elif mode == "base":
        transform_train = [transforms.Resize(size), transforms.ToTensor()]
    elif mode == "ssl":
        transform_train = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
    else:
        raise ValueError(f"{mode} mode not supported")

    if norm_layer is None:
        norm_layer = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])            
    if normalize:
        transform_train.append(norm_layer)
        transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if mode == "ssl":
        transform_train = TwoCropTransform(transform_train)

    trainset = datasets.STL10(
        root=os.path.join(data_dir, "svhn"),
        split="train",
        download=True,
        transform=transform_train,
    )

    testset = datasets.STL10(
        root=os.path.join(data_dir, "svhn"),
        split="test",
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader, norm_layer


def texture(
    data_dir, batch_size, mode="base", normalize=True, norm_layer=None, size=32
):
    """
    Minimal version since we use this dataset only for OOD evaluation.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            norm_layer,
        ]
    )
    dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "dtd/images"), transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return 0, loader, 0


# ref: https://github.com/hendrycks/outlier-exposure
def blobs(data_dir, batch_size, mode="base", normalize=True, norm_layer=None, size=32):
    """
    Minimal version since we use this dataset only for OOD evaluation.
    """
    data = np.float32(np.random.binomial(n=1, p=0.7, size=(10000, size, size, 3)))
    for i in range(10000):
        data[i] = gblur(data[i], sigma=1.5, multichannel=False)
        data[i][data[i] < 0.75] = 0.0

    dummy_targets = torch.ones(10000)
    data = torch.cat(
        [
            norm_layer(x).unsqueeze(0)
            for x in torch.from_numpy(data.transpose((0, 3, 1, 2)))
        ]
    )
    dataset = torch.utils.data.TensorDataset(data, dummy_targets)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return 0, loader, 0


def gaussian(
    data_dir, batch_size, mode="base", normalize=True, norm_layer=None, size=32
):
    """
    Minimal version since we use this dataset only for OOD evaluation.
    """
    dummy_targets = torch.ones(10000)
    ood_data = torch.from_numpy(
        np.float32(
            np.clip(
                np.random.normal(loc=0.5, size=(10000, 3, 32, 32), scale=0.25), 0, 1
            )
        )
    )
    ood_data = torch.cat([norm_layer(x).unsqueeze(0) for x in ood_data])
    dataset = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return 0, loader, 0


def uniform(
    data_dir, batch_size, mode="base", normalize=True, norm_layer=None, size=32
):
    """
    Minimal version since we use this dataset only for OOD evaluation.
    """
    dummy_targets = torch.ones(10000)
    ood_data = torch.from_numpy(
        np.float32(np.clip(np.random.uniform(size=(10000, 3, 32, 32)), 0, 1))
    )
    ood_data = torch.cat([norm_layer(x).unsqueeze(0) for x in ood_data])
    dataset = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return 0, loader, 0
