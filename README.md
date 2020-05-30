# steel-kaggle-heng
[Heng's solution](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/106462) on Kaggle [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)

## Overview
**Problem:** Predict the location and type of defects found in steel, segment each image and classify the defects.

**Data:**

- Images 256x1600
    - 12568 train
    - 1801 test (public)
- Each image may have no effects, a defect a single class, or defects of multiple classes.
- Defect masks
    - 4 classes (unknown)
    - rle encoded: submit sorted pairs of values that contain a start position and a run length, i.e. (1 3 10 5) = (1, 2, 3, 10, 11, 12, 13, 14).

**Evaluation:** Mean Dice = 2TP / (2TP + FP + FN)


**Pipeline:**

<img src="https://github.com/chinhnt0110/steel-kaggle-heng/blob/master/heng-pipeline.png" width=500 align=center>

- **Classification**
  - Model: Resnet34
  - Augment: flip, random crop, rescale, contrast, noise
  - Optimizer: SGD
  - Loss: binary cross entropy with logits
  - TTA: null, hflip, vflip
- **Segmentation**
  - Model: U-net, using Resnet18 as backbone
  - Augment: flip, random crop, rescale, contrast, noise
  - Optimizer: SGD
  - Loss: cross entropy
  - TTA: null, hflip, vflip

## Usage

The project is at `/mnt/ssd1/projects/steel-kaggle/20190910_modified`.

**Get data**
  
 - Install Kaggle API](https://github.com/Kaggle/kaggle-api) and accept competion rules.

      ``` 
      cd data
      kaggle competitions download -c kaggle competitions download -c severstal-steel-defect-detection
      unzip severstal-steel-defect-detection.zip
      ```
- Training images is already at `/mnt/ssd1/projects/steel-kaggle/steel-kaggle-heng/data/train_images` and csv file is `/mnt/ssd1/projects/steel-kaggle/steel-kaggle-heng/data/train.csv`, no need to run the above command.

**Split data**

  Split data to training set, validation set `run_make_train_split()` and test set `run_make_test_split()`
  ```
  python src/dummy_11a/kaggle.py
  ```
The classification and segmentation are trained separately.

**Train classification**
```
python src/dummy_11a/resent34_cls_01/train.py
```
Make csv for classification only:
```
python src/dummy_11a/resent34_cls_01/submit.py
```

**Train segmentation**
```
python src/dummy_11a/resnet18_unet_softmax_01/train.py
```
Make csv for segmentation only:
```
python src/dummy_11a/resnet18_unet_softmax_01/submit.py
```
**Ensemble models**

Make final csv by merging classification and segmentation:
```
python src/dummy_11a/ensemble.py
```
