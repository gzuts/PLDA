import numpy as np
import torch
import os.path as osp
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from scipy.special import comb

# from utils.gen_index_dataset import gen_index_dataset
#import synthetis_digits
import tllib.vision.datasets as datasets
import tllib.vision.models as models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.vision.datasets.imagelist import MultipleDomainsDataset
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import logging

import torchvision.datasets as torch_ds

import pickle


def load_pkl(filename):
    with open(filename, "rb") as f:
        var = pickle.load(f)
    return var


def save_pkl(filename, var):
    with open(filename, "wb") as f:
        pickle.dump(var, f)
    return


def my_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def fractional_accuracy_check(loader, model, device):
    with torch.no_grad():
        count = 0
        total, num_samples = 0, 0
        for X in loader:
            images, labels = X[0], X[1]
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
            count += 1
            if count == 20:
                break
    return total / num_samples


def accuracy_check(loader, model, device):
    with torch.no_grad():
        count = 0
        total, num_samples = 0, 0
        for X in loader:
            images, labels = X[0], X[1]
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    return total / num_samples


class gen_index_dataset(Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix
        self.true_labels = true_labels

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        each_image = self.images[index]
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]

        return each_image, each_label, each_true_label, index


def generate_uniform_cv_candidate_labels(dataname, train_labels):
    if torch.min(train_labels) > 1:
        raise RuntimeError("testError")
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = torch.max(train_labels) - torch.min(train_labels) + 1
    n = train_labels.shape[0]
    if K.int() == 65:
        cardinality = (2 ** torch.tensor(60) - 2).float()
    else:
        cardinality = (2**K - 2).float()
    number = torch.tensor(
        [comb(K, i + 1) for i in range(K - 1)]
    ).float()  # 1 to K-1 because cannot be empty or full label set, convert list to tensor
    frequency_dis = number / cardinality
    prob_dis = torch.zeros(K - 1)  # tensor of K-1
    for i in range(K - 1):
        if i == 0:
            prob_dis[i] = frequency_dis[i]
        else:
            prob_dis[i] = frequency_dis[i] + prob_dis[i - 1]

    random_n = torch.from_numpy(np.random.uniform(0, 1, n)).float()  # tensor: n
    mask_n = torch.ones(n)  # n is the number of train_data
    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0

    temp_num_partial_train_labels = 0  # save temp number of partial train_labels

    for j in range(n):  # for each instance
        for jj in range(K - 1):  # 0 to K-2
            if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                temp_num_partial_train_labels = (
                    jj + 1
                )  # decide the number of partial train_labels
                mask_n[j] = 0

        temp_num_fp_train_labels = temp_num_partial_train_labels - 1
        candidates = torch.from_numpy(
            np.random.permutation(K.item())
        ).long()  # because K is tensor type
        candidates = candidates[candidates != train_labels[j]]
        temp_fp_train_labels = candidates[:temp_num_fp_train_labels]

        partialY[j, temp_fp_train_labels] = 1.0  # fulfill the partial label matrix
    print("Finish Generating Candidate Label Sets!\n")
    return partialY


def prepare_train_loaders_for_uniform_cv_candidate_labels(
    full_train_loader, batch_size
):
    for i, (X) in enumerate(full_train_loader):
        data, labels = X[0], X[1]
        K = (
            torch.max(labels) + 1
        )  # K is number of classes, full_train_loader is full batch
    partialY = generate_uniform_cv_candidate_labels(data, labels)
    partial_matrix_dataset = gen_index_dataset(data, partialY.float(), labels.float())
    partial_matrix_train_loader = torch.utils.data.DataLoader(
        dataset=partial_matrix_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    dim = int(data.reshape(-1).shape[0] / data.shape[0])
    return partial_matrix_train_loader, data, partialY, dim


def my_get_dataset(
    dataset_name,
    root,
    source,
    target,
    train_source_transform,
    val_transform,
    train_target_transform=None,
):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name == "Digits":
        if "noise" in source[0]:
            source_name = source[0].strip("noise")
            train_source_dataset = datasets.__dict__[source_name](
                osp.join(root, source_name),
                download=True,
                split="train_noise",
                transform=train_source_transform,
            )
            if target[0] == "torchSVHN":
                train_target_transform = T.Compose(
                    [train_target_transform, T.Grayscale()]
                )
                val_transform = T.Compose([val_transform, T.Grayscale()])

                train_target_dataset = torch_ds.SVHN(
                    osp.join(root, target[0]),
                    split="train",
                    download=True,
                    transform=train_target_transform,
                )
                val_dataset = test_dataset = torch_ds.SVHN(
                    osp.join(root, target[0]),
                    split="test",
                    download=True,
                    transform=val_transform,
                )
                if len(train_target_dataset) == 1:
                    train_target_dataset = train_target_dataset[0]
                else:
                    pass
                if len(val_dataset) == 1:
                    val_dataset = val_dataset[0]
                    test_dataset = test_dataset[0]
                else:
                    pass
            elif target[0] == "torchSYND":
                train_target_transform = T.Compose(
                    [train_target_transform, T.Grayscale()]
                )
                val_transform = T.Compose([val_transform, T.Grayscale()])

                train_target_dataset = synthetis_digits.SyntheticDigits(
                    root="data",
                    download=False,
                    train=True,
                    transform=train_target_transform,
                )
                val_dataset = test_dataset = synthetis_digits.SyntheticDigits(
                    root="data", download=False, train=False, transform=val_transform
                )
                # train_target_dataset = torch_ds.SVHN(osp.join(root, target[0]), split='train',download=True, transform=train_target_transform)
                # val_dataset = test_dataset = torch_ds.SVHN(osp.join(root, target[0]), split='test',download=True, transform=val_transform )
                if len(train_target_dataset) == 1:
                    train_target_dataset = train_target_dataset[0]
                else:
                    pass
                if len(val_dataset) == 1:
                    val_dataset = val_dataset[0]
                    test_dataset = test_dataset[0]
                else:
                    pass
            else:
                train_target_dataset = datasets.__dict__[target[0]](
                    osp.join(root, target[0]),
                    download=True,
                    transform=train_target_transform,
                )
                val_dataset = test_dataset = datasets.__dict__[target[0]](
                    osp.join(root, target[0]),
                    split="test",
                    download=True,
                    transform=val_transform,
                )
            class_names = datasets.MNIST.get_classes()
            num_classes = len(class_names)
            train_source_dataset_for_validation = datasets.__dict__[source_name](
                osp.join(root, source_name), download=True, transform=val_transform
            )

        else:
            train_source_dataset = datasets.__dict__[source[0]](
                osp.join(root, source[0]),
                download=True,
                transform=train_source_transform,
            )

            if target[0] == "torchSVHN":
                train_target_transform = T.Compose(
                    [train_target_transform, T.Grayscale()]
                )
                val_transform = T.Compose([val_transform, T.Grayscale()])

                train_target_dataset = torch_ds.SVHN(
                    osp.join(root, target[0]),
                    split="train",
                    download=True,
                    transform=train_target_transform,
                )
                val_dataset = test_dataset = torch_ds.SVHN(
                    osp.join(root, target[0]),
                    split="test",
                    download=True,
                    transform=val_transform,
                )
                if len(train_target_dataset) == 1:
                    train_target_dataset = train_target_dataset[0]
                else:
                    pass
                if len(val_dataset) == 1:
                    val_dataset = val_dataset[0]
                    test_dataset = test_dataset[0]
                else:
                    pass
            elif target[0] == "torchSYND":
                train_target_transform = T.Compose(
                    [train_target_transform, T.Grayscale()]
                )
                val_transform = T.Compose([val_transform, T.Grayscale()])

                train_target_dataset = synthetis_digits.SyntheticDigits(
                    root="data",
                    download=False,
                    train=True,
                    transform=train_target_transform,
                )
                val_dataset = test_dataset = synthetis_digits.SyntheticDigits(
                    root="data", download=False, train=False, transform=val_transform
                )
                # train_target_dataset = torch_ds.SVHN(osp.join(root, target[0]), split='train',download=True, transform=train_target_transform)
                # val_dataset = test_dataset = torch_ds.SVHN(osp.join(root, target[0]), split='test',download=True, transform=val_transform )
                if len(train_target_dataset) == 1:
                    train_target_dataset = train_target_dataset[0]
                else:
                    pass
                if len(val_dataset) == 1:
                    val_dataset = val_dataset[0]
                    test_dataset = test_dataset[0]
                else:
                    pass
            else:
                train_target_dataset = datasets.__dict__[target[0]](
                    osp.join(root, target[0]),
                    download=True,
                    transform=train_target_transform,
                )
                val_dataset = test_dataset = datasets.__dict__[target[0]](
                    osp.join(root, target[0]),
                    split="test",
                    download=True,
                    transform=val_transform,
                )
            class_names = datasets.MNIST.get_classes()
            num_classes = len(class_names)
            train_source_dataset_for_validation = datasets.__dict__[source[0]](
                osp.join(root, source[0]), download=True, transform=val_transform
            )
    elif dataset_name in datasets.__dict__:
        # load datasets from tllib.vision.datasets
        dataset = datasets.__dict__[dataset_name]

        def concat_dataset(tasks, start_idx, **kwargs):
            # return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])
            return MultipleDomainsDataset(
                [dataset(task=task, **kwargs) for task in tasks],
                tasks,
                domain_ids=list(range(start_idx, start_idx + len(tasks))),
            )

        train_source_dataset = concat_dataset(
            root=root,
            tasks=source,
            download=True,
            transform=train_source_transform,
            start_idx=0,
        )
        train_target_dataset = concat_dataset(
            root=root,
            tasks=target,
            download=True,
            transform=train_target_transform,
            start_idx=len(source),
        )
        val_dataset = concat_dataset(
            root=root,
            tasks=target,
            download=True,
            transform=val_transform,
            start_idx=len(source),
        )
        if dataset_name == "DomainNet":
            test_dataset = concat_dataset(
                root=root,
                tasks=target,
                split="test",
                download=True,
                transform=val_transform,
                start_idx=len(source),
            )
        else:
            test_dataset = val_dataset
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)

        train_source_dataset_for_validation = concat_dataset(
            root=root, tasks=source, download=True, transform=val_transform, start_idx=0
        )
    else:
        raise NotImplementedError(dataset_name)
    return (
        train_source_dataset,
        train_target_dataset,
        val_dataset,
        test_dataset,
        num_classes,
        class_names,
        train_source_dataset_for_validation,
    )
