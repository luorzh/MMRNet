# MMRNet: Ensemble Deep-Learning Models for Predicting Mismatch Repair Deficiency in Endometrial Cancer from Histopathological Images

This repository contains the code for the paper "MMRNet: Ensemble Deep-Learning Models for Predicting Mismatch Repair Deficiency in Endometrial Cancer from Histopathological Images".

## Training Files
1. `train.py`: Used for training on the SYSUCC dataset, with TCGA as an external validation dataset.

## Data Preparation
1. `make_data_index.py`: Generates data indices for `train.py`.

## Mask Generation
- `mask_batch`: Generates masks. The code path needs to be modified at the end.

## Coordinate Index Generation
- `patch_coords_gen.py`: Generates coordinate indices. The path needs to be modified.

## Usage
1. Prepare the data using the data preparation scripts.
2. Generate masks and coordinate indices as needed.
3. Train the models using the training scripts.

## Citation
If you use this code, please cite our paper:
```
@article{MMRNet2025,
  title={MMRNet: Ensemble Deep-Learning Models for Predicting Mismatch Repair Deficiency in Endometrial Cancer from Histopathological Images},
  author={Li-Li Liu, Bing-Zhong Jing, Xuan Liu, Rong-Gang Li, Zhao Wan, Jiang-Yu Zhang, Xiao-Ming Ouyang, Qing-Nuan Kong, Xiao-Ling Kang, Dong-Dong Wang, Hao-Hua Chen, Zi-Han Zhao, Hao-Yu Liang, Ma-Yan Huang, Cheng-You Zheng, Xia Yang, Xue-Yi Zheng, Xin-Ke Zhang, Li-Jun Wei, Chao Cao, Hong-Yi Gao, Rong-Zhen Luo, Mu-Yan Cai},
  journal={Cell Reports Medicine},
  year={2025},
}
```


