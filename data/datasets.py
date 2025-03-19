import random
import time

import torch
from torch.utils.data.dataset import Dataset
from tqdm.std import tqdm


class TrainDataset(Dataset):
    def __init__(self, data_coords_df, slides, transform, la):
        self.data_coords_df = data_coords_df
        self.slides = slides
        self.transform = transform
        self.la = la
        self.indexs = self.data_coords_df.index.tolist()

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, item):
        index = self.indexs[item]
        row = self.data_coords_df.loc[index, :]
        wsi_name = row["wsi_name"]
        label = 1 if int(row["label"]) == 3 else 0
        la = self.la
        start_w = int(row["start_w"]) + (- (la - 1) // 2) * int(row["size_w"])
        start_h = int(row["start_h"]) + (- (la - 1) // 2) * int(row["size_w"])
        size_w = int(row["size_w"]) * la
        size_h = int(row["size_h"]) * la
        big_image = self.slides[wsi_name].read_region(
            (start_w, start_h),
            int(row["level"]),
            (size_w, size_h)
        ).convert('RGB')
        image = self.transform(big_image)  # C,H,W
        return image, label


class ValDataset(Dataset):
    def __init__(self, data_coords_df, slides, transform, la):
        self.data_coords_df = data_coords_df
        self.slides = slides
        self.transform = transform
        self.la = la
        self.indexs = self.data_coords_df.index.tolist()

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, item):
        index = self.indexs[item]
        row = self.data_coords_df.loc[index, :]
        wsi_name = row["wsi_name"]
        label = 1 if int(row["label"]) == 3 else 0
        la = self.la
        start_w = int(row["start_w"]) + (- (la - 1) // 2) * int(row["size_w"])
        start_h = int(row["start_h"]) + (- (la - 1) // 2) * int(row["size_w"])
        size_w = int(row["size_w"]) * la
        size_h = int(row["size_h"]) * la
        big_image = self.slides[wsi_name].read_region(
            (start_w, start_h),
            int(row["level"]),
            (size_w, size_h)
        ).convert('RGB')
        image = self.transform(big_image)  # C,H,W
        return index, image, label


class TestDataset(Dataset):
    def __init__(self, data_coords_df, slides, transform, la):
        self.data_coords_df = data_coords_df
        self.slides = slides
        self.transform = transform
        self.la = la
        self.indexs = self.data_coords_df.index.tolist()

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, item):
        index = self.indexs[item]
        row = self.data_coords_df.loc[index, :]
        wsi_name = row["wsi_name"]
        label = 1 if int(row["label"]) == 3 else 0
        la = self.la
        start_w = int(row["start_w"]) + (- (la - 1) // 2) * int(row["size_w"])
        start_h = int(row["start_h"]) + (- (la - 1) // 2) * int(row["size_w"])
        size_w = int(row["size_w"]) * la
        size_h = int(row["size_h"]) * la
        big_image = self.slides[wsi_name].read_region(
            (start_w, start_h),
            int(row["level"]),
            (size_w, size_h)
        ).convert('RGB')
        image = self.transform(big_image)  # C,H,W
        return index, image, label


class BalanceDataset(Dataset):
    def __init__(self, classes_datasets, data_len=None, bgs=None, weight=[]):
        self.classes_datasets = classes_datasets
        self.data_len = data_len
        self.bgs = bgs
        self.weight = weight
        self.init_bag()

    def _get_bags(self, classes_datasets):
        wsi_names = list(classes_datasets.keys())
        wsi_name_datasets = {}
        for wsi_name, dataset in classes_datasets.items():
            wsi_name_datasets[wsi_name] = list(range(len(dataset)))
            random.shuffle(wsi_name_datasets[wsi_name])
        max_lenght = (max([len(v) for v in wsi_name_datasets.values()]) // self.bgs + 1) * self.bgs
        max_lenght = max(max_lenght, self.bgs)

        for wsi_name, dataset in classes_datasets.items():
            wsi_name_dataset = list(range(len(dataset)))
            required_length = max_lenght - len(wsi_name_dataset)
            required_list = wsi_name_dataset * (required_length // len(wsi_name_dataset) + 1)
            random.shuffle(required_list)
            wsi_name_datasets[wsi_name].extend(required_list[:required_length])

        wsi_name_bags = []
        for i in range(0, max_lenght, self.bgs):
            for wsi_name in wsi_names:
                if len(wsi_name_datasets[wsi_name]) < i + self.bgs:
                    continue
                wsi_name_bags.append((wsi_name, wsi_name_datasets[wsi_name][i:i + self.bgs]))
        random.shuffle(wsi_name_bags)
        return wsi_name_bags

    def init_bag(self):
        datastes_wsi_name_bags = [self._get_bags(classes_datasets) for classes_datasets in self.classes_datasets]
        max_lenght = max([len(v) for v in datastes_wsi_name_bags])
        min_lenght = min([len(v) for v in datastes_wsi_name_bags])

        for dataste_wsi_name_bags, dataset in zip(datastes_wsi_name_bags, self.classes_datasets):
            while len(dataste_wsi_name_bags) != max_lenght:
                dataste_wsi_name_bags.extend(self._get_bags(dataset)[:max_lenght - len(dataste_wsi_name_bags)])
                print(f"{len(dataste_wsi_name_bags)}/{max_lenght}")

        self.indexs = []
        for i in tqdm(range(min_lenght)):
            for c in range(len(datastes_wsi_name_bags)):
                for j in datastes_wsi_name_bags[c][i][1]:
                    self.indexs.append((c, datastes_wsi_name_bags[c][i][0], j))

        print("train len(self.indexs)/4", len(self.indexs) / 4)

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        classes, wsi_name, index = self.indexs[item]
        return self.classes_datasets[classes][wsi_name][index]


class Rander(Dataset):
    def __init__(self, slide, locations, grids, size, rander_transform=None):
        self.slide = slide
        self.locations = locations
        self.grids = grids
        self.size = size
        self.rander_transform = rander_transform

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, item):
        grid = self.grids[item]
        location_w = self.locations[item][0]
        location_h = self.locations[item][1]
        image_patch = self.slide.read_region(
            (location_w, location_h), 0,
            (self.size, self.size)
        ).convert('RGB')
        image_patch = self.rander_transform(image_patch)
        return image_patch, grid  # (w,h)
