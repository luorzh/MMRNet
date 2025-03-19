import glob
import os
import random

import pandas as pd

from utils import get_k_fold_idx

def get_B1_sysucc_EMC_tumor_MSI(data_paths):
    type_name = "B1_sysucc_EMC_tumor_MSI"
    error_names = ["DMMR-186934_", "DMMR-186145_", "DMMR-186857_"]
    slice_dir = "Rawdata/子宫内膜癌-本中心/DMMR本中心/*.svs"
    label_dir = "appendix/B1_2023年4月14日/子宫内膜癌-本中心/DMMR本中心/*.json"
    xlsx_path = "Rawdata/internal_raw_data.xlsx"
    slice_paths = glob.glob(slice_dir)
    label_paths = glob.glob(label_dir)
    df = pd.read_excel(xlsx_path)
    count = 0
    count_ = 0
    for row in df.values:
        if row[2] == "dMMR":
            df_slice_path = row[0]
            df_label_path = row[1]
            df_slice_name = df_slice_path.split("/")[-1].split(".svs")[0]
            df_label_name = df_label_path.split("/")[-1].split(".qpdata")[0]
            flag = True
            for slice_path in slice_paths:
                slice_name = slice_path.split("/")[-1].split(".svs")[0]
                if df_slice_name == slice_name:
                    flag = False
                    break
            if flag:
                continue
            flag = True
            for label_path in label_paths:
                label_name = label_path.split("/")[-1].split("_gson.json")[0]
                if df_label_name == label_name:
                    flag = False
                    break
            if flag:
                continue
            if df_label_name not in error_names:
                data_paths[df_label_name] = {
                    "wsi_path": slice_path,
                    "label_path": label_path,
                    "type": type_name
                }
                count += 1
            count_ += 1
    print("B1_sysucc_EMC_tumor_MSI, total:", count_, "usable:", count)
    return data_paths, error_names, type_name

def get_B1_sysucc_EMC_tumor_MSS(data_paths):
    type_name = "B1_sysucc_EMC_tumor_MSS"
    error_names = []
    slice_dir = "Rawdata/子宫内膜癌-本中心/PMMR本中心/*.svs"
    label_dir = "appendix/B1_2023年4月14日/子宫内膜癌-本中心/PMMR本中心/*.json"
    xlsx_path = "Rawdata/internal_raw_data.xlsx"
    slice_paths = glob.glob(slice_dir)
    label_paths = glob.glob(label_dir)
    df = pd.read_excel(xlsx_path)
    count = 0
    count_ = 0
    for row in df.values:
        if row[2] == "pMMR":
            df_slice_path = row[0]
            df_label_path = row[1]
            df_slice_name = df_slice_path.split("/")[-1].split(".svs")[0]
            df_label_name = df_label_path.split("/")[-1].split(".qpdata")[0]
            flag = True
            for slice_path in slice_paths:
                slice_name = slice_path.split("/")[-1].split(".svs")[0]
                if df_slice_name == slice_name:
                    flag = False
                    break
            if flag:
                continue
            flag = True
            for label_path in label_paths:
                label_name = label_path.split("/")[-1].split("_gson.json")[0]
                if df_label_name == label_name:
                    flag = False
                    break
            if flag:
                continue
            if df_label_name not in error_names:
                data_paths[df_label_name] = {
                    "wsi_path": slice_path,
                    "label_path": label_path,
                    "type": type_name
                }
                count += 1
            count_ += 1
    print("B1_sysucc_EMC_tumor_MSS, total:", count_, "usable:", count)
    return data_paths, error_names, type_name

def get_B1_tcga_tumor_MSI(data_paths):
    type_name = "tcga_tumor_MSI"
    error_names = []
    slice_dir_pattern1 = "Rawdata/子宫内膜癌-其他中心/*/*.svs"
    slice_dir_pattern2 = "Rawdata/子宫内膜癌-其他中心/*/*/*.svs"
    slice_dir_pattern3 = "Rawdata/子宫内膜癌-其他中心/*/*/*/*.svs"
    label_dir_pattern1 = "appendix/B1_2023年4月14日/子宫内膜癌-其他中心/*/*.json"
    label_dir_pattern2 = "appendix/B1_2023年4月14日/子宫内膜癌-其他中心/*/*/*.json"
    label_dir_pattern3 = "appendix/B1_2023年4月14日/子宫内膜癌-其他中心/*/*/*/*.json"
    xlsx_path = "Rawdata/external_raw_data.xlsx"
    slice_paths = []
    slice_paths.extend(glob.glob(slice_dir_pattern1, recursive=True))
    slice_paths.extend(glob.glob(slice_dir_pattern2, recursive=True))
    slice_paths.extend(glob.glob(slice_dir_pattern3, recursive=True))
    label_paths = []
    label_paths.extend(glob.glob(label_dir_pattern1, recursive=True))
    label_paths.extend(glob.glob(label_dir_pattern2, recursive=True))
    label_paths.extend(glob.glob(label_dir_pattern3, recursive=True))
    df = pd.read_excel(xlsx_path)
    count = 0
    count_ = 0
    for row in df.values:
        if row[2] == "dMMR":
            df_slice_path = row[0]
            df_label_path = row[1]
            df_slice_name = df_slice_path.split("/")[-1].split(".svs")[0]
            df_label_name = df_label_path.split("/")[-1].split(".qpdata")[0]
            for slice_path in slice_paths:
                slice_name = slice_path.split("/")[-1].split(".svs")[0]
                if df_slice_name == slice_name:
                    flag = True
                    for label_path in label_paths:
                        label_name = label_path.split("/")[-1].split("_gson.json")[0]
                        if df_label_name == label_name:
                            flag = False
                            break
                    if flag:
                        continue
                    if df_label_name not in error_names:
                        if "TCGA" in slice_path:
                            data_paths[df_label_name] = {
                                "wsi_path": slice_path,
                                "label_path": label_path,
                                "type": type_name
                            }
                            count += 1
                    count_ += 1
    print("B1_tcga_tumor_MSI, total:", count_, "usable:", count)
    return data_paths, error_names, type_name

def get_B1_tcga_tumor_MSS(data_paths):
    type_name = "tcga_tumor_MSS"
    error_names = []
    slice_dir_pattern1 = "Rawdata/子宫内膜癌-其他中心/*/*.svs"
    slice_dir_pattern2 = "Rawdata/子宫内膜癌-其他中心/*/*/*.svs"
    slice_dir_pattern3 = "Rawdata/子宫内膜癌-其他中心/*/*/*/*.svs"
    label_dir_pattern1 = "appendix/B1_2023年4月14日/子宫内膜癌-其他中心/*/*.json"
    label_dir_pattern2 = "appendix/B1_2023年4月14日/子宫内膜癌-其他中心/*/*/*.json"
    label_dir_pattern3 = "appendix/B1_2023年4月14日/子宫内膜癌-其他中心/*/*/*/*.json"
    xlsx_path = "Rawdata/external_raw_data.xlsx"
    slice_paths = []
    slice_paths.extend(glob.glob(slice_dir_pattern1, recursive=True))
    slice_paths.extend(glob.glob(slice_dir_pattern2, recursive=True))
    slice_paths.extend(glob.glob(slice_dir_pattern3, recursive=True))
    label_paths = []
    label_paths.extend(glob.glob(label_dir_pattern1, recursive=True))
    label_paths.extend(glob.glob(label_dir_pattern2, recursive=True))
    label_paths.extend(glob.glob(label_dir_pattern3, recursive=True))
    df = pd.read_excel(xlsx_path)
    count = 0
    count_ = 0
    for row in df.values:
        if row[2] == "pMMR":
            df_slice_path = row[0]
            df_label_path = row[1]
            df_slice_name = df_slice_path.split("/")[-1].split(".svs")[0]
            df_label_name = df_label_path.split("/")[-1].split(".qpdata")[0]
            for slice_path in slice_paths:
                slice_name = slice_path.split("/")[-1].split(".svs")[0]
                if df_slice_name == slice_name:
                    flag = True
                    for label_path in label_paths:
                        label_name = label_path.split("/")[-1].split("_gson.json")[0]
                        if df_label_name == label_name:
                            flag = False
                            break
                    if flag:
                        continue
                    if df_label_name not in error_names:
                        if "TCGA" in slice_path:
                            data_paths[df_label_name] = {
                                "wsi_path": slice_path,
                                "label_path": label_path,
                                "type": type_name
                            }
                            count += 1
                    count_ += 1
    print("B1_tcga_tumor_MSS, total:", count_, "usable:", count)
    return data_paths, error_names, type_name


def get_train_data_index():
    data_paths = {}
    error_names = []
    type_names = []

    data_paths, error_names_, type_names_ = get_B1_sysucc_EMC_tumor_MSI(data_paths)
    error_names.extend(error_names_)
    type_names.append(type_names_)

    data_paths, error_names_, type_names_ = get_B1_sysucc_EMC_tumor_MSS(data_paths)
    error_names.extend(error_names_)
    type_names.append(type_names_)

    # Store in DataFrame
    column_names = ["wsi_name", "wsi_path", "label_path", "type"]
    rows = []
    for name in data_paths.keys():
        if name not in error_names:
            rows.append([
                name,
                data_paths[name]["wsi_path"],
                data_paths[name]["label_path"],
                data_paths[name]["type"],
            ])
    data_paths_df = pd.DataFrame(rows, columns=column_names)
    type_names = list(set(type_names))
    return data_paths_df, type_names

def get_train_fold(data_paths_df, type_names, fold=3, random_state=0):
    train_namess = []
    val_namess = []
    test_namess = []

    for type_name in list(type_names):
        samples = data_paths_df.loc[data_paths_df["type"] == type_name, "wsi_name"].values.tolist()
        _train_namess = []
        _val_namess = []
        _test_namess = []
        for train_indexs, val_indexs, test_indexs in get_k_fold_idx(samples, fold=fold, test_ratio=0, random_state=random_state):
            _train_namess.append([samples[index] for index in train_indexs])
            _val_namess.append([samples[index] for index in val_indexs])
            _test_namess.append([samples[index] for index in test_indexs])
        train_namess.append(_train_namess)
        val_namess.append(_val_namess)
        test_namess.append(_test_namess)

    val_namess_ = []
    for fold_i in range(fold):
        val_names_ = []
        for type_i in range(len(type_names)):
            val_names_.extend(val_namess[type_i][fold_i])
        val_namess_.append(val_names_)
    val_namess = val_namess_

    # Fold
    namess = val_namess
    max_len = max([len(names) for names in namess])
    new_namess = [[None] * max_len for i in range(len(namess))]
    for i, (names, new_names) in enumerate(zip(namess, new_namess)):
        for j, name in enumerate(names):
            new_namess[i][j] = name

    # Store in DataFrame
    column_names = ["fold_{}".format(f) for f in range(fold)]
    rows = []
    for names in zip(*new_namess):
        rows.append(names)

    data_index_df = pd.DataFrame(rows, columns=column_names)
    return data_index_df

def get_test_data_index():
    data_paths = {}
    error_names = []
    type_names = []

    data_paths, error_names_, type_names_ = get_B1_tcga_tumor_MSI(data_paths)
    error_names.extend(error_names_)
    type_names.append(type_names_)

    data_paths, error_names_, type_names_ = get_B1_tcga_tumor_MSS(data_paths)
    error_names.extend(error_names_)
    type_names.append(type_names_)

    # Store in DataFrame
    column_names = ["wsi_name", "wsi_path", "label_path", "type"]
    rows = []
    for name in data_paths.keys():
        if name not in error_names:
            rows.append([
                name,
                data_paths[name]["wsi_path"],
                data_paths[name]["label_path"],
                data_paths[name]["type"],
            ])
    data_paths_df = pd.DataFrame(rows, columns=column_names)
    type_names = list(set(type_names))
    return data_paths_df, type_names

def get_test_fold(data_paths_df, type_names):
    dic = {}
    max_len = 0
    for type_name in list(type_names):
        samples = data_paths_df.loc[data_paths_df["type"] == type_name, "wsi_name"].values.tolist()
        dic[type_name] = samples
        if len(samples) > max_len:
            max_len = len(samples)
    for k in dic.keys():
        l = max_len - len(dic[k])
        dic[k] = dic[k] + [None for i in range(l)]

    data_index_df = pd.DataFrame(dic)
    return data_index_df

if __name__ == "__main__":
    root = "code/EMC_MSI_cls/exp_msi_mix"
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Generate train data paths
    train_data_paths_df, train_type_names = get_train_data_index()
    train_data_paths_df_path = os.path.join(data_dir, "train_sysucc_data_paths_df.csv")
    train_data_paths_df.to_csv(train_data_paths_df_path)
    print("train_type_names: ", train_type_names)

    # Generate train cross-validation and validation set index
    data_index_df = get_train_fold(train_data_paths_df, train_type_names, fold=5, random_state=0)
    data_index_df_path = os.path.join(data_dir, "train_sysucc_data_index_df.csv")
    data_index_df.to_csv(data_index_df_path)

    # Generate test data paths
    test_data_paths_df, test_type_names = get_test_data_index()
    test_data_paths_df_path = os.path.join(data_dir, "test_tcga_data_paths_df.csv")
    test_data_paths_df.to_csv(test_data_paths_df_path)
    print("test_type_names: ", test_type_names)

    # Generate test cross-validation and validation set index
    data_index_df = get_test_fold(test_data_paths_df, test_type_names)
    data_index_df_path = os.path.join(data_dir, "test_tcga_data_index_df.csv")
    data_index_df.to_csv(data_index_df_path)
