import glob
import multiprocessing
import os
import sys
import logging
import argparse
import warnings
from copy import deepcopy

import numpy as np
import openslide
import cv2
import json
import pandas as pd
from skimage import measure
from PIL import Image
from openslide.deepzoom import DeepZoomGenerator
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from multiprocessing import Pool

from tqdm import tqdm

logging.getLogger('openslide').setLevel(logging.ERROR)
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tumor mask of tumor-WSI and save it in npy format')
parser.add_argument('--png_path', default="EMC_MSI_cls/exp_msi_0/data/mask_png", metavar='PNG_PATH', type=str, help='Path to the output npy mask file')
parser.add_argument('--RGB_min', default=30, type=int, help='min value for RGB channel, default 50')
parser.add_argument('--RGB_max', default=255, type=int, help='max value for RGB channel, default 255')
parser.add_argument('--area_min', default=40, type=int, help='minimum area for regions')
parser.add_argument('--erode_kernelSize', default=(3, 3), type=tuple, help='kernel size for erosion')
parser.add_argument('--dilate_kernelSize', default=(15, 15), type=tuple, help='kernel size for dilation')
parser.add_argument('--debug', default=False, type=bool, help='debug mode')
parser.add_argument('--cover', default=False, type=bool, help='overwrite existing files')
parser.add_argument('--num_process', default=18, type=int, help='number of processes, default 18')
parser.add_argument('--level', default=-5, type=int, help='zoom level')

def get_rgb_mask(img_RGB, RGB_min, RGB_max):
    img_HSV = rgb2hsv(img_RGB)
    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    RGB = background_R & background_G & background_B
    S = img_HSV[:, :, 1] < threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min
    max_R = img_RGB[:, :, 0] < RGB_max
    max_G = img_RGB[:, :, 1] < RGB_max
    max_B = img_RGB[:, :, 2] < RGB_max
    mask = S & RGB & min_R & min_G & min_B & max_R & max_G & max_B
    mask = mask.astype(np.uint8)
    return mask

def erode(images, kernelSize=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    eroded = [cv2.erode(image, kernel) for image in images]
    return eroded

def dilate(images, kernelSize=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    dilated = [cv2.dilate(image, kernel) for image in images]
    return dilated

def bg_mask_task(image, RGB_min, RGB_max, area_min):
    bg_mask = get_rgb_mask(image, RGB_min=RGB_min, RGB_max=RGB_max)
    def optimize_mask(mask):
        label_mask = measure.label(mask, connectivity=2)
        regions = measure.regionprops(label_mask)
        labels = set()
        for region in regions:
            if region.area < area_min:
                labels.add(region.label)
        for label in labels:
            mask[label_mask == label] = 0
        return mask
    bg_mask = optimize_mask(bg_mask) | (1 - optimize_mask(1 - bg_mask))
    return 1 - bg_mask

def get_tissue(dzSlide, level, RGB_min, RGB_max, area_min):
    w, h = dzSlide.level_dimensions[level]
    img_RGB = np.array(dzSlide.get_tile(dzSlide.level_count - 1 + level, (0, 0)).convert('RGB'))
    tissue_mask = bg_mask_task(img_RGB, RGB_min, RGB_max, area_min)
    tissue_mask = cv2.resize(tissue_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return tissue_mask

def fillPolygon(mask, chunk, value, ratio):
    original_grasp_bboxes = np.array(np.array(chunk, dtype=np.float32) * ratio, np.int32)
    original_grasp_mask = cv2.fillPoly(mask, [original_grasp_bboxes], value)
    return original_grasp_mask

def colormap(img_gray, max_label):
    img_gray = (img_gray * (255 // max_label)).astype('uint8')
    img_color = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    return img_color

def vectorgraphAnalysis(vectorgraph, maskSize, ratio=1):
    tissueTypes = []
    objectQueue = vectorgraph["childObjects"]
    for parentObject in objectQueue:
        objectQueue.extend(parentObject["childObjects"])
        tissueTypes.append(parentObject["pathObject"]["properties"]["classification"]["name"])
    tissueTypes = list(set(tissueTypes))
    masksInfo = {}
    for tissueType in tissueTypes:
        masksInfo[tissueType] = {
            'mask': np.zeros((int(maskSize[0] * ratio + 0.5), int(maskSize[1] * ratio + 0.5)), dtype=np.uint8),
            'geometrys': []
        }
    objectQueue = deepcopy(vectorgraph["childObjects"])
    for object in objectQueue:
        objectQueue.extend(object["childObjects"])
        try:
            tissueType = object["pathObject"]["properties"]["classification"]["name"]
        except:
            continue
        geometry = object["pathObject"]["geometry"]
        masksInfo[tissueType]['geometrys'].append(geometry)
    for tissueType, maskInfo in masksInfo.items():
        for geometry in maskInfo['geometrys']:
            if geometry['type'] == 'Ellipse':
                object = geometry['coordinates']
                if len(object) > 0:
                    masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=object[0], value=1, ratio=ratio)
                    for outline in object[1:]:
                        masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=outline, value=0, ratio=ratio)
            if geometry['type'] == 'Rectangel':
                object = geometry['coordinates']
                if len(object) > 0:
                    masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=object[0], value=1, ratio=ratio)
                    for outline in object[1:]:
                        masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=outline, value=0, ratio=ratio)
            if geometry['type'] == 'Polygon':
                object = geometry['coordinates']
                if len(object) > 0:
                    masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=object[0], value=1, ratio=ratio)
                    for outline in object[1:]:
                        masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=outline, value=0, ratio=ratio)
            if geometry['type'] == 'MultiPolygon':
                objects = geometry['coordinates']
                for object in objects:
                    if len(object) > 0:
                        masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=object[0], value=1, ratio=ratio)
                        for outline in object[1:]:
                            masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=outline, value=0, ratio=ratio)
    return masksInfo

def getMaskInfo(dzSlide, args):
    w, h = dzSlide.level_dimensions[-1]
    factor = 2 ** abs(args.level + 1)
    try:
        with open(args.label_path) as f:
            vectorgraph = json.load(f)
    except:
        print(f'Cannot open file or not exists.  ({args.label_path})')
    try:
        maskInfo = vectorgraphAnalysis(vectorgraph, (h, w), 1 / factor)
        w, h = dzSlide.level_dimensions[args.level]
        for tissue, maskT in maskInfo.items():
            maskT["mask"] = cv2.resize(maskT["mask"], (w, h))
        return maskInfo
    except Exception as e:
        print(e)

def getCKMask(path, size):
    img_RGB = np.array(Image.open(path))[:, :, :3]
    ck_mask = cv2.cvtColor(255 - img_RGB, cv2.COLOR_BGR2GRAY) // 255
    ck_mask = cv2.resize(ck_mask, size)
    return ck_mask

def process(opts):
    index, wsi_name, wsi_path, label_path, type, args = opts
    args.label_path = label_path
    args.wsi_path = wsi_path
    mask_path = os.path.join(args.png_path, os.path.splitext(wsi_name)[0].split("_gson")[0] + '.png')
    if not args.cover and os.path.exists(mask_path):
        return [wsi_name, mask_path]
    try:
        slide = openslide.open_slide(args.wsi_path)
        dzSlide = DeepZoomGenerator(slide, tile_size=max(slide.level_dimensions[0]), overlap=0)
        w, h = dzSlide.level_dimensions[args.level]
        tissueMask = get_tissue(dzSlide, args.level, args.RGB_min, args.RGB_max, args.area_min)
        mask = np.ones((h, w), dtype=np.uint8)
        if args.label_path:
            if "json" in args.label_path:
                mask = mask * 1
                maskInfo = getMaskInfo(dzSlide, args)
                if "Tumor" in maskInfo.keys():
                    if "MSS" in type:
                        mask[maskInfo["Tumor"]["mask"] == 1] = 2
                    if "MSI" in type:
                        mask[maskInfo["Tumor"]["mask"] == 1] = 3
        else:
            mask = mask * 1
        mask[tissueMask == 0] = 0
        max_w, max_h = slide.level_dimensions[0]
        mask = cv2.resize(mask, (max_w // 16, max_h // 16), interpolation=cv2.INTER_NEAREST)
        image = Image.fromarray(mask)
        image.save(mask_path)
        print(index, os.path.abspath(mask_path))
        return [wsi_name, mask_path]
    except Exception as e:
        warnings.warn(str(index) + ": " + str(wsi_path) + ";" + str(label_path))
        print(e)
        return [wsi_name, None]

def run(args):
    print(args)
    os.makedirs(args.png_path, exist_ok=True)
    data_paths_df = pd.read_csv("EMC_MSI_cls/exp_msi_mix/data/test_sysucc_data_paths_df.csv", index_col=0)
    opts_list = []
    for index, (wsi_name, wsi_path, label_path, type) in enumerate(data_paths_df.values):
        opts_list.append((index, wsi_name, wsi_path, label_path, type, args))
    mask_paths = []
    if args.debug:
        for (index, wsi_name, wsi_path, label_path, type, args) in opts_list:
            mask_paths.append(process((index, wsi_name, wsi_path, label_path, type, args)))
    else:
        with multiprocessing.Pool(processes=args.num_process) as pool:
            with tqdm(total=len(opts_list)) as tbar:
                for ret in pool.imap(process, opts_list):
                    mask_paths.append(ret)
                    tbar.update()
    column_names = ["wsi_name", "mask_path"]
    mask_path_df = pd.DataFrame(mask_paths, columns=column_names)
    mask_path_df.to_csv(os.path.join('EMC_MSI_cls/exp_msi_mix/data', "test_sysucc_mask_path_df.csv"))

def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
