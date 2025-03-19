import multiprocessing
import os
import warnings

from model.Resnet18Backbone import Resnet18Backbone
from model.Resnet50Backbone import Resnet50Backbone

warnings.filterwarnings('ignore', message='invalid value encountered in')
warnings.filterwarnings('ignore', message='A value is trying to be set on a copy of a slice from a DataFrame.')

from copy import deepcopy

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
print(os.path.dirname(os.path.abspath(__file__)) + '/../../')
import openslide
import argparse
import pandas as pd
from torch import optim
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from utils import setup_seed, usegpu, get_cls_index
from data.datasets import TrainDataset, ValDataset, BalanceDataset, TestDataset
from model.Head import Head

torch.multiprocessing.set_sharing_strategy("file_system")

parser = argparse.ArgumentParser(description='Train model')

parser.add_argument('--train_data_index_df_path', default='EMC_MSI_cls/exp_msi_mix/data/train_sysucc_data_index_df.csv', type=str)
parser.add_argument('--train_data_paths_df_path', default='EMC_MSI_cls/exp_msi_mix/data/train_sysucc_data_paths_df.csv', type=str)
parser.add_argument('--train_mask_path_df_path', default='EMC_MSI_cls/exp_msi_mix/data/train_sysucc_mask_path_df.csv', type=str)
parser.add_argument('--train_data_coords_df_path', default='EMC_MSI_cls/exp_msi_mix/data/train_sysucc_step4_data_coords_df.csv', type=str)
parser.add_argument('--test_data_index_df_path', default='EMC_MSI_cls/exp_msi_mix/data/test_sfy_tcga_data_index_df.csv', type=str)
parser.add_argument('--test_data_paths_df_path', default='EMC_MSI_cls/exp_msi_mix/data/test_sfy_tcga_data_paths_df.csv', type=str)
parser.add_argument('--test_mask_path_df_path', default='EMC_MSI_cls/exp_msi_mix/data/test_sfy_tcga_mask_path_df.csv', type=str)
parser.add_argument('--test_data_coords_df_path', default='EMC_MSI_cls/exp_msi_mix/data/test_sfy_tcga_step8_data_coords_df.csv', type=str)

parser.add_argument('--save_path', default="EMC_MSI_cls/exp_msi_0/result2/baseline117|AdamW|lr3e-4|wd1e-5|train_sysucc_step8_data_coords_df|test_sfy_tcga_step8_data_coords_df|[ALL, TCGA]|bs(128+64)*4|bgs(128+64)*4*4|grad_accu8|resnet18|", metavar='SAVE_PATH', type=str, help='Path to the saved models')

parser.add_argument('--name_list', default=["ALL"])
parser.add_argument('--model_path')
parser.add_argument('--backbone', default="Resnet18Backbone")
parser.add_argument('--opti', default="AdamW")
parser.add_argument('--grad_accu', default=2, type=int, help='grad_accu')
parser.add_argument('--batch_size', type=int, default=(8) * 8 * 4, help='mini-batch size (default: 512)')
parser.add_argument('--bgs', type=int, default=(8) * 1 * 4, help='mini-batch size (default: 512)')
parser.add_argument('--train_data_len', default=10000 * (8) * 1 * 4, type=int, help='number of epochs')
parser.add_argument('--nepochs', type=int, default=10, help='number of epochs')
parser.add_argument('--workers', default=8 * 4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=1, type=int, help='test on val every (default: 10)')
parser.add_argument('--lr', default=3e-3, type=float, help='learn rate (default: 1e-4)')
parser.add_argument('--wd', default=1e-5, type=float, help='weight_decay (default: 1e-4)')
parser.add_argument('--gpu_ids', default="0,1,2,3", type=str, help='choise gpu id (default: 0,1,2,3')
parser.add_argument('--seed', default=0, type=int, help='choise seed (default: 123)')
parser.add_argument('--train_limit_num', default=None, type=int, help='Small-scale train (default: None)')
parser.add_argument('--test_limit_num', default=None, type=int, help='Small-scale test (default: None)')
parser.add_argument('--la', default=1, type=int, help='Small-scale test (default: None)')

args = parser.parse_args()
def main(args):
    # Load dataframes
    train_data_index_df = pd.read_csv(args.train_data_index_df_path, index_col=0)
    train_data_paths_df = pd.read_csv(args.train_data_paths_df_path, index_col=0)
    train_mask_path_df = pd.read_csv(args.train_mask_path_df_path, index_col=0)
    train_data_coords_df = pd.read_csv(args.train_data_coords_df_path, index_col=0, low_memory=False)
    test_data_index_df = pd.read_csv(args.test_data_index_df_path, index_col=0)
    test_data_paths_df = pd.read_csv(args.test_data_paths_df_path, index_col=0)
    test_mask_path_df = pd.read_csv(args.test_mask_path_df_path, index_col=0)
    test_data_coords_df = pd.read_csv(args.test_data_coords_df_path, index_col=0, low_memory=False)

    # Build slide cache
    slides = {}
    for wsi_name, wsi_path in tqdm(train_data_paths_df.loc[:, ["wsi_name", "wsi_path"]].drop_duplicates().values, desc="Build slides..."):
        try:
            slides[wsi_name] = openslide.OpenSlide(wsi_path)
        except:
            print(wsi_path)
    for wsi_name, wsi_path in tqdm(test_data_paths_df.loc[:, ["wsi_name", "wsi_path"]].drop_duplicates().values, desc="Build slides..."):
        try:
            slides[wsi_name] = openslide.OpenSlide(wsi_path)
        except:
            print(wsi_path)

    all_fold = len([v for v in train_data_index_df.columns.values if 'fold' in v])
    for fold_i in range(0, train_data_index_df.values.shape[1]):
        save_path = os.path.join(args.save_path, str(fold_i))
        os.makedirs(save_path, exist_ok=True)
        args.start_epoch = 0

        # Initialize model
        macenko_norm = MacenkoNorm(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        if args.backbone == "Resnet18Backbone":
            backbone = Resnet18Backbone(in_channels=3 * args.la * args.la, out_channels=256, pretrained=True)
        elif args.backbone == "Resnet50Backbone":
            backbone = Resnet50Backbone(in_channels=3 * args.la * args.la, out_channels=256, pretrained=True)
        head = Head(in_channels=256, num_classes=2)
        loss_sce_fn = torch.nn.CrossEntropyLoss()
        if args.use_gpu:
            macenko_norm = macenko_norm.cuda()
            backbone = backbone.cuda()
            head = head.cuda()
            macenko_norm = torch.nn.DataParallel(macenko_norm)
            backbone = torch.nn.DataParallel(backbone)
            head = torch.nn.DataParallel(head)
            loss_sce_fn = loss_sce_fn.cuda()

        params_list = [
            {'params': backbone.parameters()},
            {'params': head.parameters()},
        ]
        if args.opti == "AdamW":
            optimizer = optim.AdamW(params_list, lr=args.lr, weight_decay=args.wd)
        elif args.opti == "SGD":
            optimizer = optim.SGD(params_list, lr=args.lr)
        optimizer.zero_grad()

        # Load checkpoint if provided
        if args.model_path is not None:
            ckpt = torch.load(args.model_path.format(str(fold_i)))
            macenko_norm.load_state_dict(ckpt["macenko_norm"])
            backbone.load_state_dict(ckpt["backbone"])
            head.load_state_dict(ckpt["head"])
            optimizer.load_state_dict(ckpt["optimizer"])
            args.start_epoch = ckpt["epoch"] + 1

        # Define transformations
        norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=512 * args.la, scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.4),
            transforms.RandomGrayscale(p=0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Generate training and validation datasets
        train_names = []
        val_names = []
        for i in range(all_fold):
            if i != fold_i:
                train_names.extend(train_data_index_df["fold_{}".format(i)].values.tolist())
            else:
                val_names.extend(train_data_index_df["fold_{}".format(i)].values.tolist())

        test_names = {type_name: test_data_index_df[type_name].values.tolist() for type_name in test_data_index_df.columns.values.tolist()}

        train_type_names = train_data_paths_df["type"].unique().tolist()
        val_type_names = train_data_paths_df["type"].unique().tolist()
        test_type_names = test_data_paths_df["type"].unique().tolist()

        train_type_names_dict = {train_type_name: train_data_paths_df.loc[train_data_paths_df["type"].apply(lambda x: x == train_type_name), "wsi_name"].values.tolist() for train_type_name in train_type_names}
        val_type_names_dict = {val_type_name: train_data_paths_df.loc[train_data_paths_df["type"].apply(lambda x: x == val_type_name), "wsi_name"].values.tolist() for val_type_name in val_type_names}
        test_type_names_dict = {test_type_name: test_data_paths_df.loc[test_data_paths_df["type"].apply(lambda x: x == test_type_name), "wsi_name"].values.tolist() for test_type_name in test_type_names}

        dataset_type_names = {
            "train": {train_type_name: train_data_paths_df.loc[train_data_paths_df["wsi_name"].apply(lambda x: (x in train_names) and (x in train_type_names_dict[train_type_name])), "wsi_name"].tolist() for train_type_name in train_type_names},
            "val": {val_type_name: train_data_paths_df.loc[train_data_paths_df["wsi_name"].apply(lambda x: (x in val_names) and (x in val_type_names_dict[val_type_name])), "wsi_name"].tolist() for val_type_name in val_type_names},
            "test": {test_type_name: test_data_paths_df.loc[test_data_paths_df["wsi_name"].apply(lambda x: (x in test_names[test_type_name]) and (x in test_type_names_dict[test_type_name])), "wsi_name"].tolist() for test_type_name in test_type_names},
        }

        data_coords_dfs = {
            "train": {train_type_name: train_data_coords_df.loc[train_data_coords_df.loc[:, "wsi_name"].apply(lambda x: x in dataset_type_names["train"][train_type_name]), :] for train_type_name in train_type_names},
            "val": {val_type_name: train_data_coords_df.loc[train_data_coords_df.loc[:, "wsi_name"].apply(lambda x: x in dataset_type_names["val"][val_type_name]), :] for val_type_name in val_type_names},
            "test": {test_type_name: test_data_coords_df.loc[test_data_coords_df.loc[:, "wsi_name"].apply(lambda x: x in dataset_type_names["test"][test_type_name]), :] for test_type_name in test_type_names}
        }

        data_coords_dfs_class = {
            "train": {train_type_name: {2: data_coords_dfs["train"][train_type_name].loc[(data_coords_dfs["train"][train_type_name]["label"] == 2)], 3: data_coords_dfs["train"][train_type_name].loc[(data_coords_dfs["train"][train_type_name]["label"] == 3)]} for train_type_name in train_type_names},
            train_data_paths_df["type"].apply(lambda x: x == val_type_name), "wsi_name"].values.tolist() for
                               val_type_name in val_type_names}
        test_type_names_dict = {test_type_name: test_data_paths_df.loc[
            test_data_paths_df["type"].apply(lambda x: x == test_type_name), "wsi_name"].values.tolist() for
                                test_type_name in test_type_names}

        dataset_type_names = {
            "train": {train_type_name: train_data_paths_df.loc[train_data_paths_df["wsi_name"].apply(
                lambda x: (x in train_names) and (x in train_type_names_dict[train_type_name])), "wsi_name"].tolist()
                      for train_type_name in train_type_names},
            "val": {val_type_name: train_data_paths_df.loc[train_data_paths_df["wsi_name"].apply(
                lambda x: (x in val_names) and (x in val_type_names_dict[val_type_name])), "wsi_name"].tolist() for
                    val_type_name in val_type_names},
            "test": {test_type_name: test_data_paths_df.loc[test_data_paths_df["wsi_name"].apply(
                lambda x: (x in test_names[test_type_name]) and (
                        x in test_type_names_dict[test_type_name])), "wsi_name"].tolist() for test_type_name in
                     test_type_names},
        }
        data_coords_dfs = {
            "train": {train_type_name: train_data_coords_df.loc[train_data_coords_df.loc[:, "wsi_name"].apply(
                lambda x: x in dataset_type_names["train"][train_type_name]), :] for train_type_name in
                      train_type_names},
            "val": {val_type_name: train_data_coords_df.loc[train_data_coords_df.loc[:, "wsi_name"].apply(
                lambda x: x in dataset_type_names["val"][val_type_name]), :] for val_type_name in val_type_names},
            "test": {test_type_name: test_data_coords_df.loc[test_data_coords_df.loc[:, "wsi_name"].apply(
                lambda x: x in dataset_type_names["test"][test_type_name]), :] for test_type_name in test_type_names}
        }
        data_coords_dfs_class = {
            "train": {train_type_name: {
                2: data_coords_dfs["train"][train_type_name].loc[
                    (data_coords_dfs["train"][train_type_name]["label"] ==2)],
                3: data_coords_dfs["train"][train_type_name].loc[
                    (data_coords_dfs["train"][train_type_name]["label"] == 3)],
            } for train_type_name in train_type_names},
            "val": {val_type_name:{
                2: data_coords_dfs["val"][val_type_name].loc[
                    (data_coords_dfs["val"][val_type_name]["label"] ==2)],
                3: data_coords_dfs["val"][val_type_name].loc[
                    (data_coords_dfs["val"][val_type_name]["label"] == 3)],
            }  for val_type_name in val_type_names},
            "test": {test_type_name: {
                2: data_coords_dfs["test"][test_type_name].loc[
                    (data_coords_dfs["test"][test_type_name]["label"] ==2)],
                3: data_coords_dfs["test"][test_type_name].loc[
                    (data_coords_dfs["test"][test_type_name]["label"] == 3)],
            } for test_type_name in test_type_names},
        }
        datasets = {
            "train":{
                "B1_sysucc_EMC_tumor_ALL": {
                    2: {wsi_name:
                        TrainDataset(df , slides, train_transform, args.la)
                        for wsi_name,df in pd.concat([
                            data_coords_dfs_class["train"]["B1_sysucc_EMC_tumor_MSS"][2],
                        ], axis=0).groupby('wsi_name')
                    },
                    3: {wsi_name:
                        TrainDataset(df , slides, train_transform, args.la)
                        for wsi_name,df in pd.concat([
                            data_coords_dfs_class["train"]["B1_sysucc_EMC_tumor_MSI"][3],
                        ], axis=0).groupby('wsi_name')
                    },
                },
            },
            "val": {
                "B1_sysucc_EMC_tumor_ALL": {
                    2: {wsi_name:
                        ValDataset(df, slides, val_transform,args.la)
                        for wsi_name, df in data_coords_dfs_class["val"]["B1_sysucc_EMC_tumor_MSS"][2].groupby('wsi_name')
                    },
                    3: {wsi_name:
                        ValDataset(df, slides, val_transform,args.la)
                        for wsi_name, df in data_coords_dfs_class["val"]["B1_sysucc_EMC_tumor_MSI"][3].groupby('wsi_name')
                    },
                },

            },
            "test": {
                "tumor_ALL": TestDataset(pd.concat([
                    data_coords_dfs_class["test"]["tcga_EMC_tumor_MSS"][2],
                    data_coords_dfs_class["test"]["tcga_EMC_tumor_MSI"][3],
                    data_coords_dfs_class["test"]["sfy_tumor_MSS"][2],
                    data_coords_dfs_class["test"]["sfy_tumor_MSI"][3]
                ],axis=0), slides,test_transform,args.la),
            }
        }



        # train
        train_dataset = BalanceDataset(
            classes_datasets=[
                datasets["train"]["B1_sysucc_EMC_tumor_ALL"][2],
                datasets["train"]["B1_sysucc_EMC_tumor_ALL"][3],
            ],
            # data_len=500
            data_len=args.train_data_len,
            bgs = args.bgs,
            weight=[0.5, 0.5]
        )  # 234421


        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,

        )

        val_loaders = {}
        for val_dataset_name in datasets["val"].keys():
            val_loaders[val_dataset_name] = DataLoader(
                BalanceDataset(
                    classes_datasets=[
                        datasets["val"][val_dataset_name][2],
                        datasets["val"][val_dataset_name][3],
                    ],

                    data_len=args.train_data_len,
                    bgs=args.bgs,
                    weight=[0.5, 0.5]
                ),
                batch_size=args.batch_size//2,
                shuffle=True,
                num_workers=args.workers//2,
                pin_memory=False,
                drop_last=True

            )
        test_loaders = {}

        for test_dataset_name in datasets["test"].keys():
            test_loaders[test_dataset_name] = DataLoader(
                datasets["test"][test_dataset_name],
                batch_size=args.batch_size//2,
                shuffle=True,
                num_workers=args.workers//4,
                pin_memory=False,
                drop_last=True
            )

        # 开始训练
        val_best_auc = 0
        test_best_auc = 0
        with SummaryWriter(os.path.join(save_path, "log")) as summary_writer:
            for epoch in range(args.start_epoch, args.nepochs):

                # 初始化bag
                train_loader.dataset.init_bag()
                # 训练
                train(fold_i, epoch, backbone, head, train_loader, loss_sce_fn, optimizer, summary_writer, save_path,"Train",args.la)
                # 设置学习率

                torch.save({
                    'epoch': epoch,
                    'backbone': backbone.state_dict(),
                    'head': head.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(save_path, "train_best_epoch_{}.ckpt".format(epoch)))
                #
                if epoch % args.test_every == 0 or epoch == args.nepochs - 1:
                    # 验证

                    for val_dataset_name in list(val_loaders.keys()):
                        val_crc_nef_auc,result_coords_df = test(fold_i, epoch, backbone, head, val_loaders[val_dataset_name], loss_sce_fn, summary_writer,train_data_coords_df,train_data_paths_df, save_path,"Val {}".format(val_dataset_name),args.la)

                        result_coords_df.to_csv(os.path.join(args.save_path, str(fold_i), 'log','val_{}_result_epoch{}.csv'.format(val_dataset_name,epoch)))

                        if val_crc_nef_auc >= val_best_auc:
                            val_best_auc = val_crc_nef_auc
                            torch.save({
                                'epoch': epoch,
                                'backbone': backbone.state_dict(),
                                'head': head.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'crc_nef_auc':val_crc_nef_auc,
                            }, os.path.join(save_path, "val_best_epoch_{}_{}.ckpt".format(epoch,val_dataset_name)))
                    # 测试
                    for test_dataset_name in list(test_loaders.keys()):
                        test_crc_nef_auc,result_coords_df = test(fold_i, epoch, backbone, head, test_loaders[test_dataset_name], loss_sce_fn, summary_writer,test_data_coords_df,test_data_paths_df, save_path,"Test {}".format(test_dataset_name),args.la)

                        result_coords_df.to_csv(os.path.join(args.save_path, str(fold_i), 'log','test_{}_result_epoch{}.csv'.format(test_dataset_name,epoch)))
                        if test_crc_nef_auc >= test_best_auc:
                            test_best_auc = test_crc_nef_auc
                            torch.save({
                                'epoch': epoch,
                                'backbone': backbone.state_dict(),
                                'head': head.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'crc_nef_auc': test_crc_nef_auc,
                            }, os.path.join(save_path, "test_best_epoch_{}.ckpt".format(epoch)))



def train(fold_i, epoch, backbone, head, loader, loss_sce_fn, optimizer, summary_writer, save_path,dataset_name,la):
    preds_list = []
    probs_list = []
    label_list = []
    loss_list = []
    backbone.train()
    head.train()

    for step, (image, label) in enumerate(tqdm(loader, desc="{} epoch:{}...".format(dataset_name,epoch))):
        shuffled_indices = torch.randperm(len(image))
        image = image[shuffled_indices]
        label = label[shuffled_indices]
        # print(label)
        if args.use_gpu:
            image = image.contiguous().cuda()
            label = label.contiguous().cuda()


        # 堆叠
        B,C, H, W = image.shape
        image = image.reshape((B,C, la, H // la, la, W // la)).permute(0,2, 4, 1, 3, 5).reshape(B,C * la * la, H // la,
                                                                                                W // la)


        encode= backbone(image)
        out = head(encode)

        loss = loss_sce_fn(out, label)

        # bp
        loss = loss / args.grad_accu
        loss.backward()
        if step % args.grad_accu == 0 or step == len(loader) - 1:
            optimizer.step()
            optimizer.zero_grad()
        # 记录

        # 记录
        prob = torch.nn.functional.softmax(out.detach(), dim=1)[:, 1]
        pred = torch.argmax(torch.nn.functional.softmax(out.detach(), dim=1), dim=1)
        probs_list.append(prob.detach().cpu())
        preds_list.append(pred.detach().cpu())
        label_list.append(label.detach().cpu())
        loss_list.append(loss.detach().cpu())

    probs = torch.cat(probs_list, dim=-1).numpy()
    preds = torch.cat(preds_list, dim=-1).numpy()
    label = torch.cat(label_list, dim=-1).numpy()
    losses = torch.tensor(loss_list).numpy()
    acc, sensitivity, specificity, ppv, npv, f1_score, auc = get_cls_index(label, preds, probs)
    index_std = np.array(([sensitivity, specificity, ppv, npv])).std()
    loss = losses.mean()
    result_str = "{} epoch: {}\n".format(dataset_name,epoch)
    result_str += '模型\t权重\t阈值\tINDEX方差\tAUC\tACC\tF1_Score\tSensitivity\tSpecificity\tPPV\tNPV\n'
    result_str += "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n".format(
        fold_i,
        "未知",
        round(0.5, 4),
        round(index_std, 4),
        round(auc, 4),
        round(acc, 4),
        round(f1_score, 4),
        round(sensitivity, 4),
        round(specificity, 4),
        round(ppv, 4),
        round(npv, 4)
    )
    print(result_str)

    # 文档记录
    with open(os.path.join(save_path, 'log/result.log'), 'a+') as f:
        f.write(result_str)
    # 画曲线
    summary_writer.add_scalar("{}/epoch/loss".format(dataset_name), loss, epoch)

    summary_writer.add_scalar("{}/epoch/acc".format(dataset_name), acc, epoch)
    summary_writer.add_scalar("{}/epoch/sensitivity".format(dataset_name), sensitivity, epoch)
    summary_writer.add_scalar("{}/epoch/specificity".format(dataset_name), specificity, epoch)
    summary_writer.add_scalar("{}/epoch/ppv".format(dataset_name), ppv, epoch)
    summary_writer.add_scalar("{}/epoch/npv".format(dataset_name), npv, epoch)
    summary_writer.add_scalar("{}/epoch/f1_score".format(dataset_name), f1_score, epoch)
    summary_writer.add_scalar("{}/epoch/auc".format(dataset_name), auc, epoch)


def test(fold_i, epoch, backbone, head, loader, loss_sce_fn, summary_writer,data_coords_df,data_paths_df, save_path,dataset_name,la):
    idxs_list = []
    preds_list = []
    probs_list = []
    label_list = []
    loss_list = []

    backbone.eval()
    head.eval()
    with torch.no_grad():
        for step, (idx,image, label) in enumerate(tqdm(loader, desc="{} epoch:{}...".format(dataset_name,epoch))):

            if args.use_gpu:
                image = image.cuda()
                label = label.cuda()

            B, C, H, W = image.shape
            image = image.reshape((B, C, la, H // la, la, W // la)).permute(0, 2, 4, 1, 3, 5).reshape(B,
                                                                                                      C * la * la,
                                                                                                      H // la,
                                                                                                      W // la)

            encode = backbone(image)
            out = head(encode)

            loss = loss_sce_fn(out, label)

            # 记录
            idxs_list.append(idx.detach().cpu())
            prob = torch.nn.functional.softmax(out.detach(), dim=1)[:, 1]
            pred = torch.argmax(torch.nn.functional.softmax(out.detach(), dim=1), dim=1)
            probs_list.append(prob.detach().cpu())
            preds_list.append(pred.detach().cpu())
            label_list.append(label.detach().cpu())
            loss_list.append(loss.detach().cpu())

        idxs = torch.cat(idxs_list, dim=-1).numpy()
        probs = torch.cat(probs_list, dim=-1).numpy()
        preds = torch.cat(preds_list, dim=-1).numpy()
        labels = torch.cat(label_list, dim=-1).numpy()
        losses = torch.tensor(loss_list).numpy()
        acc, sensitivity, specificity, ppv, npv, f1_score, auc = get_cls_index(labels, preds, probs)
        index_std = np.array(([sensitivity, specificity, ppv, npv])).std()
        loss = losses.mean()
        result_str = "{} epoch: {}\n".format(dataset_name,epoch)
        result_str += '模型\t权重\t阈值\tINDEX方差\tAUC\tACC\tF1_Score\tSensitivity\tSpecificity\tPPV\tNPV\n'
        result_str += "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n".format(
            fold_i,
            "未知",
            round(0.5, 4),
            round(index_std, 4),
            round(auc, 4),
            round(acc, 4),
            round(f1_score, 4),
            round(sensitivity, 4),
            round(specificity, 4),
            round(ppv, 4),
            round(npv, 4)
        )
        print(result_str)
        # 文档记录
        with open(os.path.join(save_path, 'log/result.log'), 'a+') as f:
            f.write(result_str)
        # 画曲线
        summary_writer.add_scalar("{}/epoch/loss".format(dataset_name), loss, epoch)

        summary_writer.add_scalar("{}/epoch/acc".format(dataset_name), acc, epoch)
        summary_writer.add_scalar("{}/epoch/sensitivity".format(dataset_name), sensitivity, epoch)
        summary_writer.add_scalar("{}/epoch/specificity".format(dataset_name), specificity, epoch)
        summary_writer.add_scalar("{}/epoch/ppv".format(dataset_name), ppv, epoch)
        summary_writer.add_scalar("{}/epoch/npv".format(dataset_name), npv, epoch)
        summary_writer.add_scalar("{}/epoch/f1_score".format(dataset_name), f1_score, epoch)
        summary_writer.add_scalar("{}/epoch/auc".format(dataset_name), auc, epoch)
        # slide
        result_df = pd.DataFrame({'index': idxs, 'prob': probs, 'labels': labels}).set_index('index')
        result_coords_df = pd.merge(data_coords_df.iloc[idxs, :], result_df, left_index=True, right_index=True)
        result_coords_df = result_coords_df.sort_index()

        slide_result_coords_df = result_coords_df.groupby('wsi_name').mean()
        slide_result_coords_df = slide_result_coords_df.reset_index()
        slide_result_coords_df = slide_result_coords_df.rename(columns={"prob": "slide_prob"})
        result_coords_df = pd.merge(result_coords_df, slide_result_coords_df.loc[:, ["wsi_name","slide_prob"]], on='wsi_name')
        # patch prob， slide prob  slide label
        if "Test" in dataset_name:
            name_list=args.name_list
        else:
            name_list=["ALL"]
        for name in name_list:

            if name=="ALL":
                selected_rows=result_coords_df
            elif name == "Other":

                selected_name = np.unique(data_paths_df.loc[data_paths_df["wsi_path"].apply(lambda x: len([otherName for otherName in args.otherNames if otherName in x])>0), "wsi_name"].values)
                selected_rows = result_coords_df.loc[result_coords_df["wsi_name"].isin(selected_name)]
            else:
                selected_name = np.unique(data_paths_df.loc[data_paths_df["wsi_path"].apply(lambda x: name in x), "wsi_name"].values)
                selected_rows = result_coords_df.loc[result_coords_df["wsi_name"].isin(selected_name)]

            best_threshold = 0.5

            selected_rows['predicted_labels'] = np.where(selected_rows['slide_prob'] > best_threshold, 1, 0)

            selected_rows = selected_rows[['wsi_name', 'labels', 'predicted_labels', 'slide_prob']]
            selected_rows = selected_rows.drop_duplicates()

            slide_prob = selected_rows.loc[:, "slide_prob"].values
            slide_pred = (slide_prob > 0.5).astype(np.float32)
            slide_label = selected_rows.loc[:, "labels"].values

            acc_slide, sensitivity_slide, specificity_slide, ppv_slide, npv_slide, f1_score_slide, auc_slide = get_cls_index(slide_label, slide_pred, slide_prob)
            index_std_slide = np.array(([sensitivity_slide, specificity_slide, ppv_slide, npv_slide])).std()
            result_str = "{}...{} slide epoch: {}\n".format(dataset_name,name, epoch)
            result_str += '模型\t权重\t阈值\tINDEX方差\tAUC\tACC\tF1_Score\tSensitivity\tSpecificity\tPPV\tNPV\n'
            result_str += "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n".format(
                fold_i,
                "未知",
                round(best_threshold, 4),
                round(index_std_slide, 4),
                round(auc_slide, 4),
                round(acc_slide, 4),
                round(f1_score_slide, 4),
                round(sensitivity_slide, 4),
                round(specificity_slide, 4),
                round(ppv_slide, 4),
                round(npv_slide, 4)
            )
            print(result_str)

            # 文档记录
            with open(os.path.join(save_path, 'log/result.log'), 'a+') as f:
                f.write(result_str)
            # 画曲线
            summary_writer.add_scalar("{}/epoch/best_threshold".format("{}: {}".format(dataset_name,name)), best_threshold, epoch)
            summary_writer.add_scalar("{}/epoch/acc_slide".format("{}: {}".format(dataset_name,name)), acc_slide, epoch)
            summary_writer.add_scalar("{}/epoch/sensitivity_slide".format("{}: {}".format(dataset_name,name)), sensitivity_slide, epoch)
            summary_writer.add_scalar("{}/epoch/specificity_slide".format("{}: {}".format(dataset_name,name)), specificity_slide, epoch)
            summary_writer.add_scalar("{}/epoch/ppv_slide".format("{}: {}".format(dataset_name,name)), ppv_slide, epoch)
            summary_writer.add_scalar("{}/epoch/npv_slide".format("{}: {}".format(dataset_name,name)), npv_slide, epoch)
            summary_writer.add_scalar("{}/epoch/f1_score_slide".format("{}: {}".format(dataset_name,name)), f1_score_slide, epoch)
            summary_writer.add_scalar("{}/epoch/auc_slide".format("{}: {}".format(dataset_name,name)), auc_slide, epoch)
    return auc,result_coords_df


if __name__ == '__main__':
    args.use_gpu = usegpu(args.gpu_ids)
    setup_seed(args.seed)
    main(args)
