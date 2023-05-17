# 4220 Final Project

**The report is `report.pdf`.**

## Installation

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**Step 1.** Install MMSegmentation.

```
pip install -v -e .
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

## Training

In this project, We mainly conduct four experiments.

1. FCN+U-Net

   ```
   python /tools/train.py my_config/fcn_unet.py --work-dir fcn
   ```

2. Deeplabv3+

   ```
   python /tools/train.py my_config/deeplabv3plus.py --work-dir deeplabv3plus
   ```

3. Deeplabv3+ with pretraining

   ```
   python /tools/train.py my_config/deeplabv3plus_pretrain.py --work-dir deeplabv3plus_pretrain
   ```

4. Deeplabv3+ with pretraining and Lovasz loss

   ```
   python /tools/train.py my_config/deeplabv3plus_pretrain_lovasz.py --work-dir deeplabv3plus_pretrain_lovasz
   ```

## Unzip the checkpoint

   The best-performanced checkpoint is stored in `checkpoints/iter_2000.zip`. Since Github has a limited file size of 100mb, the checkpoint is zipped into 4 files. 

## Testing

Since we only upload the model with the best performance, only the third experiment has the checkpoint.

1. FCN+U-Net

   ```
   python /tools/test.py my_config/fcn_unet.py {checkpoint} --work-dir fcn
   ```

2. Deeplabv3+

   ```
   python /tools/train.py my_config/deeplabv3plus.py {checkpoint} --work-dir deeplabv3plus
   ```

3. Deeplabv3+ with pretraining

   ```
   python /tools/train.py my_config/deeplabv3plus_pretrain.py checkpoints/iter_2000.pth --work-dir deeplabv3plus_pretrain
   ```

4. Deeplabv3+ with pretraining and Lovasz loss

   ```
   python /tools/train.py my_config/deeplabv3plus_pretrain_lovasz.py {checkpoint} --work-dir deeplabv3plus_pretrain_lovasz
   ```

## Inferencing

```
python inference.py
```

Please see details in `inference.py`.
