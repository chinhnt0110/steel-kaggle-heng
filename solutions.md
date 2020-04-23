#Severstal: Steel Defect Detection

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

**Userful libs:**
- [segmentation_model_pytorch library](https://github.com/qubvel/segmentation_models.pytorch)
###1st place solution
*Classification*


*Segmentation*


###55th place solution
[Github](https://github.com/khornlund/severstal-steel-defect-detection)

Best submission

| Public LB     | Private LB    | 
| :------------:|:-------------:|
| 0.91817       | 0.91023       |
- Model:
    - Encoder: `Efficientnet-b5`, `InceptionV4`, `se_resnext50_32x4d`
    - Decoder: Unet (with Dropout) + FPN

- Loss = `(0.6 * BCE) + (0.4 * (1 - Dice))`
- Optimizer: RAdam
- Augmentation: `Normalize`, `Flip`, `RandomCrop`, `CropNonEmptyMaskIfExists`, `GaussianBlur`, `IAASharpen`.
- Pseudo Labeling:
- Post-processing:

**Classification Model**

scaling pixel thresholds due to classifier output

**Final Ensemble**
- Unet
    - 2x seresnext5032x4d
    - 1x efficientnet-b5
- FPN
    - 3x seresnext5032x4d
    - 1x efficientnet-b5
    - 1x inceptionv4
**Improvement**

[See more](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114410)
###3rd place solution

Best submission

| Public LB     | Private LB    | 
| :------------:|:-------------:|
| 0.918https    | 0.90934       |
Pipeline
![Pipeline](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F263239%2Ff7bf0093fee5274cf1aa07efc29b0c7a%2Fmodel.png?generation=1573784178983559&alt=media)
[Discussion]()
###4th place solution
###5th place solution
###7th place solution
###8th place solution
Only segmentation, no classifier.

Models: Ensemble FPN-B0, B1, B2, B3, B4, Seresnext50. UNET-Seresnext50, Resnet34. Custom Attention Unet-B0, B1.

Augmentation: Flipping, Random Brightness, Random Gamma, Random Contrast.

Freeze encoder for fast convergence, only batchnorm unfrozen.

Post-processing: Triplet threshold, [reference](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107824#latest-650999).
###9th place solution
Intuition: dependent

Why need to filter out empty masks?

"The leaderboard score is the mean of the Dice coefficients for each ImageId, ClassId pair in the test set."

It means that score in this competiton was calculated separately for each image and for each defect type. So if ground 
truth is mask with zero pixels and you will predict zero pixels - you have score 1, but if you predict at least one 
pixel - your score is 0.


###10th place solution
###11th place solution
###12th place solution
[Discussion](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114309)

Intuition

Pipeline

Images -> Stage1 -> Filter FP -> Stage2 -> Filter FP again -> Predict Mask

Stage 1. Multilabel classification

Model:
* Best 3 of 5 folds Senet154 trained on resized images (128x800) with simple BCE loss

Augmentations: Normalization, resize, h-flip, v-flip, no TTA

Stage 2. Multilabel segmentation

Stage 3. Binary segmentation
###13th place solution
###30th place solution
[Github](https://github.com/bamps53/kaggle-severstal)
###31th place solution
[Github](https://github.com/Diyago/Severstal-Steel-Defect-Detection)


[loss is stuck](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/108270)