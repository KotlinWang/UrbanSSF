# Accurate Segmentation of Urban Spatial Structure: A Framework for Large-Scale Remote Sensing Images Using Feature State Sequences

https://github.com/user-attachments/assets/df3c2974-1717-4e07-b301-4f6274e5ecd8.mp4

## :newspaper:News

- **[2024/8/4]** UrbanSSF Project Creation :sunglasses:. 

## :star:Overview

![overview](./assets/urbanssf.jpg)
- UrbanSSF is the first to combine CNNs, Transformers and Mamba for the remote sensing of VHR urban scenes. The Global Semantic Enhancer (GSE) module and the Spatial Interactive Attention (SIA) mechanism process different scale features from the encoder. FSI Mamba uses the powerful sequence modeling ability of state space module to apply to the feature state sequence. The Channel Space Reconstruction (CSR) algorithm is designed to reduce the computational complexity of large-scale feature fusion.
- UrbanSSF has achieved the effect of SOTA on three urban scene datasets of UAVid, ISPRS Vaihingen and Potsdam. Especially on the UAVid dataset.

##  :dart:Model Zoo

| **Dateset**         | **Method**     | **Params(M)↓** | **FPS↑** | **mIoU↑** | Download |
| :------------------------ | :------------: | :-------------: | :-----------------: | :---------------: | :---------------: |
| **Vaihingen**       | UrbanSSF-T | **3.6** | **66.3**        | 83.3          | [model](https://drive.google.com/file/d/1cpvsf6bIml_NZ8ouFIx9SpBL0CL5zFI1/view?usp=drive_link) |
|                           | UrbanSSF-S | 14.0   | 25.6        | 84.5          | [model](https://drive.google.com/file/d/1iLU7PioDTnvzuBdcbueLyepp0uyCWRQl/view?usp=drive_link) |
|                           | UrbanSSF-L | 60.0     | 10.2           | **85.0**     | [model](https://drive.google.com/file/d/13G7285_lCU_lhi51T-zehfL_oragNsy-/view?usp=drive_link) |
| **Potsdam**     | UrbanSSF-T |    **3.6**     | **66.3** | 85.4          | [model](https://drive.google.com/file/d/1S4sSC_Xp3YjWSwEOfmm6yvp6VnIM_XXD/view?usp=drive_link) |
|               | UrbanSSF-S |      14.0      |   25.6   | 86.9          | [model](https://drive.google.com/file/d/1cMcxlzT3ajtLJvN5PWbDCpUBF__GOMRa/view?usp=drive_link) |
|               | UrbanSSF-L |      60.0      |   10.2   | **87.6**        | [model](https://drive.google.com/file/d/1f5oHB72AWyWyCXV3Cjd6UFGEHnxc8Vft/view?usp=drive_link) |
| **UAVid**            | UrbanSSF-T |    **3.6**     | **66.3** | 65.7         | [model](https://drive.google.com/file/d/1Rl88F1Ooetvk1r527jDmhdNLYgTe8BuB/view?usp=drive_link) |
|               | UrbanSSF-S |      14.0      |   25.6   | 69.8          | [model](https://drive.google.com/file/d/1AlE_0PcB4PDwrevA86PZOAyeloH8tHvE/view?usp=drive_link) |
|               | UrbanSSF-L |      60.0      |   10.2   | **71.0**      | [model](https://drive.google.com/file/d/1TCrxbzjV907jBYI1AsDQBuZsE5FR6DgU/view?usp=drive_link) |

## :see_no_evil:Visualization

##### UAVid
<div align="center">
<img src="./assets/uavid.jpg" height="80%" width="80%" />
</div>

##### Vaihingen
<div align="center">
<img src="./assets/vaihingen.jpg" height="80%" width="80%" />
</div>

##### Potsdam
<div align="center">
<img src="./assets/potsdam.jpg" height="80%" width="80%" />
</div>

## :computer:Installation

<details open>

**Step 0**: Clone this project and create a conda environment:

   ```shell
   git clone https://github.com/KotlinWang/UrbanSSF.git
   cd UrbanSSF
   
   conda create -n urbanssf python=3.11
   conda activate urbanssf
   ```

**Step 1**: Install pytorch and torchvision matching your CUDA version:

   ```shell
   pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
   ```

**Step 2**: Install requirements:

   ```shell
   pip install -r requirements.txt
   ```

**Step 3**: Install Mamba:

   ```shell
   pip install causal-conv1d==1.2.0.post2
   
   pip install mamba-ssm==1.2.0.post1
   ```

</details>

## :satellite:Dataset Preparation

<details open>

Download the [ISPRS Vaihingen, Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspxdatasets) and [UAVid](https://uavid.nl/) dateset.

**Vaihingen**

Generate the training set.
```shell
python tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/train_images" \
--mask-dir "data/vaihingen/train_masks" \
--output-img-dir "data/vaihingen/train/images_1024" \
--output-mask-dir "data/vaihingen/train/masks_1024" \
--mode "train" --split-size 1024 --stride 512 
```
Generate the testing set.
```shell
python tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/test_images" \
--mask-dir "data/vaihingen/test_masks_eroded" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \
--eroded
```
Generate the masks_1024_rgb (RGB format ground truth labels) for visualization.

````shell
python tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/test_images" \
--mask-dir "data/vaihingen/test_masks" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024_rgb" \
--mode "val" --split-size 1024 --stride 1024 \
--gt
````

**Potsdam**
````shell
python tools/potsdam_patch_split.py \
--img-dir "data/potsdam/train_images" \
--mask-dir "data/potsdam/train_masks" \
--output-img-dir "data/potsdam/train/images_1024" \
--output-mask-dir "data/potsdam/train/masks_1024" \
--mode "train" --split-size 1024 --stride 1024 --rgb-image 
`````
As for the validation set, you can select some images from the training set to build it.

````shell
python tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks_eroded" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \
--eroded --rgb-image
````

```shell
python tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024_rgb" \
--mode "val" --split-size 1024 --stride 1024 \
--gt --rgb-image
```

**UAVid**
```shell
python tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train_val" \
--output-img-dir "data/uavid/train_val/images" \
--output-mask-dir "data/uavid/train_val/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

```shell
python tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train" \
--output-img-dir "data/uavid/train/images" \
--output-mask-dir "data/uavid/train/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

```shell
python tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_val" \
--output-img-dir "data/uavid/val/images" \
--output-mask-dir "data/uavid/val/masks" \
--mode 'val' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

</details>

## :running: Training

"-c" means the path of the config, use different **config** to train different models.

```shell
python train_supervision.py -c config/uavid/unetformer.py
```

## :mag: Testing

"-c" denotes the path of the config, Use different **config** to test different models. 

"-o" denotes the output path 

"--rgb" denotes whether to output masks in RGB format

**Vaihingen**
```
python vaihingen_test.py -c config/vaihingen/urbanssf-s.py -o fig_results/vaihingen/urbanssf-s --rgb -t 'None'
```

**Potsdam**

```
python potsdam_test.py -c config/potsdam/urbanssf-s.py -o fig_results/potsdam/urbanssf-s --rgb -t 'None'
```

**UAVid**

```
python uavid_test.py -c config/uavid/urbanssf-s.py -o fig_results/uavid/urbanssf-s --rgb -t 'None'
```

## Acknowledgement

- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [UNetFormer](https://github.com/WangLibo1995/GeoSeg)
- [Vision Mamba](https://github.com/hustvl/Vim)

## Citation

If you find this project useful in your research, please consider citing：

```
```

