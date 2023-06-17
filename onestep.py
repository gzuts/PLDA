import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import loss as loss_func
import numpy as np
import random
import network
from torch.utils.data import DataLoader
import pandas as pd
import utils
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from RCCC_utils.utils_algo import confidence_update
from RCCC_utils.utils_loss import rc_loss
import tllib_da_utils.utils as tl_utils
from tllib.alignment.dann import ImageClassifier
from datetime import datetime

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.autograd.set_detect_anomaly(True)


def Tsharpen(input):
    lamda = 0.5
    sharp = torch.pow(input, 1 / lamda)
    sharp2 = torch.sum(sharp, 1)
    result = torch.div(sharp, sharp2.view(-1, 1))
    return result


def train(args, model, ad_net, train_source_loader,
        train_target_loader, optimizer, optimizer_ad, epoch, method, confidence):
    model.train()
    confidence = confidence.cuda()

    len_source = len(train_source_loader)
    len_target = len(train_target_loader)
    if len_source > len_target:
        num_iter = len_source
    else:
        num_iter = len_target

    for batch_idx in range(num_iter):
        if batch_idx % len_source == 0:
            iter_source = iter(train_source_loader)
        if batch_idx % len_target == 0:
            iter_target = iter(train_target_loader)

        ad_net.train()
        images, labels, true_labels, index = iter_source.next()
        data_source, label_source, true_labels, index = (
            images.cuda(),
            labels.cuda(),
            true_labels.cuda(),
            index.cuda(),
        )

        target_tuple = iter_target.next()
        data_target = target_tuple[0]
        data_target = data_target.cuda()

        optimizer.zero_grad()
        optimizer_ad.zero_grad()

        output, feature = model(data_source)
        loss = rc_loss(output, confidence, index)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        confidence = confidence_update(
            model, confidence, data_source, label_source, index
        )

        output, feature = model(torch.cat((data_source, data_target)))
        softmax_output = nn.Softmax(dim=1)(output)

        if method == "CDAN-E":
            softmax_output = Tsharpen(softmax_output)
            entropy = loss_func.Entropy(softmax_output)
            loss2 = loss_func.CDAN(
                [feature, softmax_output],
                ad_net,
                entropy,
                network.calc_coeff(num_iter * (epoch) + batch_idx),
                None,
            )

        elif method == "CDAN-noT":
            entropy = loss_func.Entropy(softmax_output)
            loss2 = loss_func.CDAN(
                [feature, softmax_output],
                ad_net,
                entropy,
                network.calc_coeff(num_iter * (epoch) + batch_idx),
                None,
            )

        elif method == "CDAN-T":
            entropy = loss_func.Entropy(softmax_output)
            loss2 = loss_func.CDAN(
                [feature, softmax_output], ad_net, None, None, random_layer
            )

        elif method == "CDAN":
            loss2 = loss_func.CDAN(
                [feature, softmax_output], ad_net, None, None, random_layer
            )

        elif method == "DANN":
            ad_out = ad_net(feature)
            batch_size = ad_out.size(0) // 2
            dc_target = (
                torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size))
                .float()
                .cuda()
            )
            loss2 = args.trade_off * nn.BCELoss()(ad_out, dc_target)

        elif method == "tl-DANN":
            loss2 = args.trade_off * domain_adv(
                feature[: len(data_source)], feature[len(data_source) :]
            )
        else:
            raise ValueError("Method cannot be recognized.")

        loss2.backward()
        optimizer.step()
        optimizer_ad.step()

    return confidence, loss


def main():
    parser = argparse.ArgumentParser(description="CDAN USPS MNIST")
    parser.add_argument("root", metavar="DIR", help="root path of dataset")
    parser.add_argument(
        "-d",
        "--data",
        metavar="DATA",
        default="Office31",
        help="dataset: "
        + " | ".join(tl_utils.get_dataset_names())
        + " (default: Office31)",
    )
    parser.add_argument("--method", type=str, default="DANN")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1000,
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=550,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr_ad",
        type=float,
        default=0.005,
        metavar="LR2",
        help="learning rate2 (default: 0.01)",
    )
    parser.add_argument(
        "--momentum_ad",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--wd_ad",
        "--weight-decay_ad",
        default=1e-3,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-3)",
        dest="weight_decay_ad",
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="cuda device id")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--random", type=bool, default=False, help="whether to use random"
    )
    parser.add_argument(
        "--pretrain", type=bool, default=True, help="whether to use random"
    )

    parser.add_argument(
        "-mo", help="model name", default="resnet18", type=str, required=False
    )

    parser.add_argument("-s", "--source", help="source domain(s)", nargs="+")
    parser.add_argument("-t", "--target", help="target domain(s)", nargs="+")
    parser.add_argument("--train-resizing", type=str, default="default")
    parser.add_argument("--val-resizing", type=str, default="default")
    parser.add_argument(
        "--resize-size", type=int, default=32, help="the image size after resizing"
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs="+",
        default=[0.08, 1.0],
        metavar="PCT",
        help="Random resize scale (default: 0.08 1.0)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        nargs="+",
        default=[3.0 / 4.0, 4.0 / 3.0],
        metavar="RATIO",
        help="Random resize aspect ratio (default: 0.75 1.33)",
    )
    parser.add_argument(
        "--no-hflip",
        action="store_true",
        help="no random horizontal flipping during training",
    )
    parser.add_argument(
        "--norm-mean",
        type=float,
        nargs="+",
        default=(0.485, 0.456, 0.406),
        help="normalization mean",
    )
    parser.add_argument(
        "--norm-std",
        type=float,
        nargs="+",
        default=(0.229, 0.224, 0.225),
        help="normalization std",
    )
    parser.add_argument(
        "--task",
        help="task of domain adatation",
        default="none",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--no-pool",
        action="store_true",
        help="no pool layer after the feature extractor.",
    )
    parser.add_argument(
        "--bottleneck-dim", default=256, type=int, help="Dimension of bottleneck"
    )
    parser.add_argument(
        "--scratch", action="store_true", help="whether train from scratch."
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.01,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--lr-gamma", default=0.001, type=float, help="parameter for lr scheduler"
    )
    parser.add_argument(
        "--lr-decay", default=0.75, type=float, help="parameter for lr scheduler"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "-td",
        "--trade_off",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-3,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-3)",
        dest="weight_decay",
    )

    parser.add_argument("--model_path", help="whether train from scratch.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    ###############
    # Prepare Data
    ###############
    train_transform = tl_utils.get_train_transform(
        args.train_resizing,
        scale=args.scale,
        ratio=args.ratio,
        random_horizontal_flip=not args.no_hflip,
        random_color_jitter=False,
        resize_size=args.resize_size,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )
    val_transform = tl_utils.get_val_transform(
        args.val_resizing,
        resize_size=args.resize_size,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    (
        train_source_dataset,
        train_target_dataset,
        val_dataset,
        test_dataset,
        num_classes,
        args.class_names,
        train_source_dataset_for_validation,
    ) = utils.my_get_dataset(
        args.data, args.root, args.source, args.target, train_transform, val_transform
    )
    train_source_loader = DataLoader(
        train_source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    train_target_loader = DataLoader(
        train_target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    train_data_size = len(train_source_dataset)
    full_train_source_loader = DataLoader(
        train_source_dataset,
        batch_size=train_data_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    (
        partial_train_source_loader,
        train_data,
        train_givenY,
        dim,
    ) = utils.prepare_train_loaders_for_uniform_cv_candidate_labels(
        full_train_loader=full_train_source_loader, batch_size=args.batch_size
    )

    train_source_loader_for_validation = DataLoader(
        train_source_dataset_for_validation,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )

    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
    confidence = train_givenY.float() / tempY
    backbone = tl_utils.get_model(args.mo, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    model = ImageClassifier(
        backbone,
        num_classes,
        bottleneck_dim=args.bottleneck_dim,
        pool_layer=pool_layer,
        finetune=not args.scratch,
    ).to(device)

    model = model.cuda()
    class_num = num_classes

    if args.method == "DANN":
        ad_net = network.AdversarialNetwork(args.bottleneck_dim, 500)
    elif args.method == "CDAN":
        ad_net = network.AdversarialNetwork(args.bottleneck_dim * class_num, 500)
    elif args.method == "CDAN-E":
        ad_net = network.AdversarialNetwork(args.bottleneck_dim * class_num, 500)
    elif args.method == "CDAN-noT":
        ad_net = network.AdversarialNetwork(args.bottleneck_dim * class_num, 500)
    elif args.method == "CDAN-T":
        ad_net = network.AdversarialNetwork(args.bottleneck_dim * class_num, 500)

    ad_net = ad_net.cuda()

    optimizer_ad = optim.SGD(
        ad_net.get_parameters(),
        lr=args.lr_ad,
        weight_decay=args.weight_decay_ad,
        momentum=args.momentum_ad,
    )
    acc_best = 0
    optimizer = SGD(
        model.get_parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    lr_scheduler = LambdaLR(
        optimizer,
        lambda x: args.lr * (1.0 + args.lr_gamma * float(x)) ** (-args.lr_decay),
    )

    exp_prefix = (
        "ours_"
        + args.data
        + "_"
        + args.source[0]
        + "to"
        + args.target[0]
        + "_"
        + args.method
    )
    exp_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    exp_name = exp_prefix + "-" + exp_datetime

    logpath = os.path.join("logs", exp_name + ".log")
    logger = utils.my_logger(logpath)

    logger.info("start training!")
    logger.info(args)

    writer = SummaryWriter(log_dir=os.path.join("tensorboard_logs", exp_name))
    best_model = None

    target_acc_list = []

    for epoch in range(1, args.epochs + 1):
        confidence, loss = train(
            args,
            model,
            ad_net,
            partial_train_source_loader,
            train_target_loader,
            optimizer,
            optimizer_ad,
            epoch,
            args.method,
            confidence,
        )

        model.eval()
        test_accuracy_target = utils.accuracy_check(
            loader=test_loader, model=model, device=device
        )
        acc = test_accuracy_target
        if acc >= acc_best:
            acc_best = acc
            best_model = model

        logger.info(
            "Epoch: {}. Target Te Acc: {}.".format(epoch + 1, test_accuracy_target)
            + "\n"
        )
        writer.add_scalars(
            "Test_target_accuracy", {"acc_target:": acc}, global_step=epoch
        )
        target_acc_list.append(test_accuracy_target)
        lr_scheduler.step()

    logger.info("Best target Te Acc: {}.".format(acc_best))
    torch.save(model, "saved_models/" + exp_name + ".model")

    test_acc = pd.DataFrame(columns=["test_acc"], data=target_acc_list)
    test_acc.to_csv("test_acc_logs/" + exp_name + ".csv", encoding="gbk")


if __name__ == "__main__":
    main()
