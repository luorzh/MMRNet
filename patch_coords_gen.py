import argparse
import glob
import multiprocessing
import os
import random
import sys
import warnings
from multiprocessing.pool import Pool
from random import shuffle
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tqdm.std import tqdm

sys.path.append(os.path.join(os.path.abspath(__file__), "/../../"))

parser = argparse.ArgumentParser(description="Get center points of patches from mask")
parser.add_argument("--mask_path_df_path", type=str)
parser.add_argument("--coords_path", type=str)
parser.add_argument('--num_process', default=36, type=int, help='number of multi-processes, default 8')
parser.add_argument("--patch_size", default=(224)//16, type=int)
parser.add_argument("--patch_step", default=(224)//16, type=int)
parser.add_argument('--debug', default=False, type=bool, help='')
parser.add_argument("--seed", default=0, type=int)

def getDirContent(dirPath, fileType, isJoin):
    files_name = []
    files_name_temp = os.listdir(dirPath)
    if fileType == 'file':
        for file_name_temp in files_name_temp:
            if os.path.isfile((os.path.join(dirPath, file_name_temp))):
                files_name.append(file_name_temp)
    elif fileType == 'dir':
        for file_name_temp in files_name_temp:
            if not os.path.isfile((os.path.join(dirPath, file_name_temp))):
                files_name.append(file_name_temp)
    if isJoin:
        files_path = [os.path.join(dirPath, file_name) for file_name in files_name]
        return files_path
    return files_name

def list_shuffle_split(ls, ratio):
    lss = []
    while len(ls) != 0:
        shuffle(ls)
        lt = int(ratio[0] * len(ls) + 0.5)
        lss.append(ls[0:lt])
        ls = ls[lt+1:]
        ratio = [v / sum(ratio[1:]) for i, v in enumerate(ratio) if i != 0]
    return lss

def process(opts):
    try:
        id, wsi_name, mask_path, patch_size, patch_step = opts
        print(id, wsi_name, mask_path)
        mask = cv2.imread(mask_path)
        padding = [(mask.shape[0] % patch_step) // 2, (mask.shape[1] % patch_step) // 2]
        centers = []
        count = 0
        all_count = 0
        for row in range(0, mask.shape[0], patch_step):
            for col in range(0, mask.shape[1], patch_step):
                patch_mask = mask[row:row+patch_size, col:col+patch_size, 0]
                label = None
                if patch_mask.__contains__(2):
                    label = 2
                if patch_mask.__contains__(3):
                    label = 3
                if label is not None:
                    k = (patch_mask == label).mean()
                    if k > 0.8:
                        centers.append([wsi_name, label, 0, col * 16, row * 16, patch_size * 16, patch_size * 16])
                        count += 1
                all_count += 1
        print(wsi_name, "{}/{}".format(count, all_count))
        column_names = ["wsi_name", "label", 'level', "start_w", "start_h", "size_w", "size_h"]
        data_coords_df = pd.DataFrame(centers, columns=column_names)
        return data_coords_df
    except Exception as e:
        warnings.warn(str(id) + ": " + str(mask_path))
        print(e)
        column_names = ["wsi_name", "mask_path", "center_w", "center_h"]
        data_coords_df = pd.DataFrame([], columns=column_names)
        return data_coords_df

def run(args):
    patch_size = args.patch_size
    patch_step = args.patch_step
    mask_path_df = pd.read_csv(args.mask_path_df_path, index_col=0)
    mask_path_df = mask_path_df.loc[mask_path_df["mask_path"].notnull(), :]
    opts_list = []
    for id, (wsi_name, mask_path) in enumerate(mask_path_df.values):
        opts_list.append((id, wsi_name, mask_path, patch_size, patch_step))
        print(wsi_name)
    data_coords_dfs = []
    with multiprocessing.Pool(processes=args.num_process) as pool:
        with tqdm(total=len(opts_list)) as tbar:
            for ret in pool.imap(process, opts_list):
                data_coords_dfs.append(ret)
                tbar.update()
    data_coords_df = pd.concat(data_coords_dfs, axis=0).reset_index(drop=True)
    data_coords_df.to_csv(args.coords_path)
    print(len(data_coords_df))

if __name__ == "__main__":
    args = parser.parse_args()
    random.seed(args.seed)
    print(args)
    run(args)
