# Severstal: Steel Defect Detection

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
- [segmentation_model_pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [albumentations](https://github.com/albumentations-team/albumentations)
- [mlcomp](https://github.com/catalyst-team/mlcomp)+[catalyst](https://www.kaggle.com/lightforever/severstal-mlcomp-catalyst-infer-0-90672)
## 1st place solution
**Classification**
- Filter out images with no defects.
- Train data: 224x1568 random crop images
- Augmentations: Randomcrop, Hflip, Vflip, RandomBrightnessContrast (from albumentations) and a customized defect blackout.
- Batchsize: 8 for efficientnet-b1, 16 for resnet34 (both accumulate gradients for 32 samples).
- Optimizer: SGD
- Model Ensemble: 3 x efficientnet-b1+1 x resnet34
- TTA: None, Hflip, Vflip
- Threshold: 0.6,0.6,0.6,0.6

**Segmentation**
- Train data: 256x512 crop images
- Augmentations: Hflip, Vflip, RandomBrightnessContrast.
- Batchsize: 12 or 24 (both accumulate gradients for 24 samples)
- Optimizer: Rectified Adam
- Models: Unet (efficientnet-b3), FPN (efficientnet-b3).
- Loss: BCE (with posweight = (2.0,2.0,1.0,1.5)) 0.75BCE+0.25DICE (with posweight = (2.0,2.0,1.0,1.5))
- Model Ensemble: 1 x Unet(BCE loss) + 3 x FPN(first trained with BCE loss then finetuned with BCEDice loss) +2 x FPN(BCEloss)+ 3 x Unet
- TTA: None, Hflip, Vflip
- Label Thresholds: 0.7, 0.7, 0.6, 0.6
- Pixel Thresholds: 0.55,0.55,0.55,0.55
- Postprocessing: Remove whole mask if total pixel < threshold (600,600,900,2000) + remove small components with size <150

**Pseudo Label**

The pseudo labels are chosen if classifiers and segmentation networks make the same decisions.

An image will only be chosen if the probabilities from classifiers are all over 0.95 or below 0.05 and it gets same result from segmentation part. According to this rule, 1135 images are chosen and added to trainset.
## 55th place solution
[Github](https://github.com/khornlund/severstal-steel-defect-detection)

Best submission

| Public LB     | Private LB    | 
| :------------:|:-------------:|
| 0.91817       | 0.91023       |
- Model:
    - Encoder: `Efficientnet-b5`, `InceptionV4`, `se_resnext50_32x4d`
    - Decoder: Unet (with Dropout) + FPN

- Loss = `(0.6 * BCE) + (0.4 * (1 - Dice))`
- Optimizer: Rectified Adam
- Augmentation: `Normalize`, `Flip`, `RandomCrop`, `CropNonEmptyMaskIfExists`, `GaussianBlur`, `IAASharpen`.
- Pseudo Labeling:
- Post-processing:

**Classification Model**

Scaling pixel thresholds due to classifier output.

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
## 3rd place solution
Best submission

| Public LB     | Private LB    | 
| :------------:|:-------------:|
| 0.91824       | 0.90934       |

Pipeline

![Pipeline](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F263239%2Ff7bf0093fee5274cf1aa07efc29b0c7a%2Fmodel.png?generation=1573784178983559&alt=media)
- Basic Model: Unet, Feature Pyramid Network (FPN)
- Encoder: efficientnet-b3, efficientnet-b4, efficientnet-b5, se-resnext50
- Loss: Focal Loss
- Optimizer: Adam, init lr = 0.0005
- Learning Rate Scheduler: ReduceLROnPlateau (factor=0.5, patience=3,cooldown=3, min_lr=1e-8)
- Image Size: 256x800 for training, 256x1600 for inference
- Image Augmentation: horizontal flip, vertical flip
- Sampler: Weighted Sampler
- Ensemble Model: average 9 model output probability to achieve the final mask probability without TTA.
    1. FPN + efficientnet-b5 + concatenation of feature maps
    2. FPN + efficientnet-b4
    3. Unet + efficientnet-b4 , add pseudo labeling data in training data
    4. Unet + efficientnet-b4, training with heavy image augmentation
    5. Unet + efficientnet-b4 +SCSE layer
    6. Unet + efficientnet-b4 +SCSE layer, add pseudo labeling data in training data
    7. Unet + efficientnet-b4 + Mish layer
    8. Unet + efficientnet-b3
    9. Unet + se-resnext50
    
[See more](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/117377)
## 4th place solution
## 5th place solution
## 7th place solution
## 8th place solution
Only segmentation, no classifier.

Models: Ensemble FPN-B0, B1, B2, B3, B4, Seresnext50. UNET-Seresnext50, Resnet34. Custom Attention Unet-B0, B1.

Augmentation: Flipping, Random Brightness, Random Gamma, Random Contrast.

Freeze encoder for fast convergence, only batchnorm unfrozen.

Post-processing: Triplet threshold, [reference](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107824#latest-650999).
## 9th place solution
Intuition: dependent

Why need to filter out empty masks?

"The leaderboard score is the mean of the Dice coefficients for each ImageId, ClassId pair in the test set."

It means that score in this competiton was calculated separately for each image and for each defect type. So if ground 
truth is mask with zero pixels and you will predict zero pixels - you have score 1, but if you predict at least one 
pixel - your score is 0.


## 10th place solution
## 11th place solution
## 12th place solution
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

## 13th place solution
**Classification**
- Model: 3x EfficientNet-b4 (first 3 of stratified 10 folds)
- Input: full size 
- Augmentation: random crop rescale, hflip, vflip, random contrast, random gamma, random brightness
- TTA: none, hflip
- Threshold label: 

**Segmentation**
- Models
    - EfficientNet-b3 Unet stratified 4fold w/ full size image
    - EfficientNet-b3 Unet 1fold w/ random crop 256 x 800
    - 3x Unet
- Augmentation: random crop rescale, hflip, vflip, random contrast, random gamma, random brightness
- Loss: BCEDice (bce weiht=0.75, dice weight=0.25)
- TTA: none, hflip
- Threshold mask
- Postprocess

[See more](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114332)

## 30th place solution
[Github](https://github.com/bamps53/kaggle-severstal)

Apply 5 class classification including background class and then 4 class segmentation.

- Models: 
    - resnet50, efficient-b3, resnext50 for classification
    - Unet with resnet18, PSPNet with resnet18, FPN with resnet50 for segmentation
- Loss: BCE + Dice loss
- Augmentation: Random crop by 256x800 or full 256x1600
## 31th place solution
[Github](https://github.com/Diyago/Severstal-Steel-Defect-Detection)
- Models: FPN + se-resnext50
- Augmentation: `Hflip`, `Vflip`, `RandomBrightnessContrast`, crop while training then fine tune with full size.
- Loss: BCE with dice, Lovasz loss
- Optimizer: Adam with RAdam
- Sampler: BalanceClassSampler
- Pseudo labling: 2 rounds: by training and validating.
- Post-processing: Filling holes, remove small masks by threshold
- Ensemble: average different encoders

[loss is stuck](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/108270)
