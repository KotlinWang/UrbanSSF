# Accurate Segmentation of Urban Spatial Structure: A Framework for Large-Scale Remote Sensing Images Using Feature State Sequences

https://github.com/user-attachments/assets/df3c2974-1717-4e07-b301-4f6274e5ecd8.mp4

## :newspaper:News

- **[2024/8/4]** UrbanSSF Project Creation :sunglasses:. 

## :star:Overview

![overview](./assets/urbanssf.jpg)
- UrbanSSF is the first to combine CNNs, Transformers and Mamba for the remote sensing of VHR urban scenes. The Global Semantic Enhancer (GSE) module and the spatial interactive attention (SIA) mechanism process different scale features from the encoder. FSI Mamba uses the powerful sequence modeling ability of state Space Module (SSMs) to apply to the feature state sequence. Channel Space Reconstruction (CSR) algorithm is designed to reduce the computational complexity of large-scale feature fusion.
- UrbanSSF has achieved the effect of SOTA on three urban scene datasets of UAVid, ISPRS Vaihingen and Potsdam. Especially on the UAVid dataset.

## :bookmark_tabs: Main Result

| **Type**                  | **Method**     | **Backbone** | **UAVid mIoU↑** | **Vaihingen OA↑** | **Vaihingen mF1↑** | **Vaihingen mIoU↑** | **Potsdam OA↑** | **Potsdam mF1↑** | **Potsdam mIoU↑** |
| ------------------------- | -------------- | ------------ | --------------- | ----------------- | ------------------ | ------------------- | --------------- | ---------------- | ----------------- |
| **CNNs**                  | BiSeNet        | ResNet-18    | 61.5            | 87.1              | 84.3               | 75.8                | 88.2            | 89.8             | 81.7              |
|                           | DANet          | ResNet-18    | 60.6            | 88.2              | 79.6               | 69.4                | 89.1            | 88.9             | 80.3              |
|                           | ShelfNet       | ResNet-18    | 47.0            | 89.8              | 87.5               | 78.3                | 89.9            | 91.3             | 84.4              |
|                           | FANet          | ResNet-18    | -               | 88.9              | 85.4               | 75.6                | 89.8            | 91.3             | 84.2              |
|                           | EaNet          | ResNet-18    | -               | 89.7              | 87.7               | 78.7                | 88.7            | 90.6             | 83.4              |
|                           | SwiftNet       | ResNet-18    | 61.1            | 90.2              | 88.3               | 79.6                | 89.3            | 91.0             | 83.8              |
|                           | MAResU-Net     | ResNet-18    | -               | 90.1              | 87.7               | 78.6                | 89.0            | 90.5             | 83.9              |
|                           | ABCNet         | ResNet-18    | 63.8            | 90.7              | 89.5               | 81.3                | 90.7            | 91.9             | 85.2              |
|                           | PACSCNet       | Res2Net-50   | -               | 90.0              | -                  | 82.3                | 85.2            | -                | 76.0              |
| **Transformer**           | TransUNet      | ViT-R50      | -               | -                 | 79.9               | 67.1                | -               | 85.4             | 75.0              |
|                           | Segmenter      | ViT-Tiny     | 58.7            | 88.1              | 84.1               | 73.6                | 88.7            | 89.2             | 80.7              |
|                           | BANet          | ResT-Lite    | 64.6            | 90.5              | 89.6               | 81.4                | 91.0            | 92.5         | 86.3          |
|                           | BoTNet         | ResNet-18    | 63.2            | 88.0              | 84.8               | 74.3                | -               | -                | -                 |
|                           | UNetFormer     | ResNet-18    | 66.1        | 91.0              | 90.4           | 82.7            | 90.8            | 92.0             | 85.3              |
|                           | SwinUNet       | Swin-Tiny    | -               | -                 | 72.0               | 58.0                | -               | 78.8             | 65.5              |
|                           | ST-UNet        | ResNet-50    | -               | -                 | 82.2               | 70.2                | -               | 86.1             | 76.0              |
|                           | AerialFormer-S | Swin-B       | -               | -                 | -                  | -                   | 91.3        | 87.2             | 79.3              |
| **Mamba**                 | Mamba-UNet     | VSS          | 57.3            | 92.6          | 89.7               | 81.6                | 88.9            | 90.1             | 82.3              |
|                           | Swin-UMamba    | VSS          | 53.4            | 92.4              | 89.4               | 81.3                | 89.1            | 90.4             | 82.7              |
|                           | VM-UNet        | VSS          | 55.7            | 92.3              | 88.3               | 79.6                | 88.2            | 89.3             | 80.9              |
|                           | Samba          | Samba        | -               | -                 | -                  | 73.6                | -               | -                | 82.3              |
| **CNN+Transformer+Mamba** | UrbanSSF-T     | MobilNet-V3  | 65.7            | 93.1              | 90.7               | 83.3                | 90.9            | 92.0             | 85.4              |
|                           | UrbanSSF-S     | RegNet-Y     | 69.8            | 93.3              | 91.4               | 84.5               | 91.7   | 92.9    | 86.9     |
|                           | UrbanSSF-L     | ResNeXt-101  | **71.0**        | **93.6**          | **91.7**           | **85.0**            | **92.2**  | **93.3**   | **87.6**    |
| **Improvement**           |                |              | **+4.9**        | **+1.0**          | **+1.3**           | **+2.3**            | **+0.9**        | **+0.8**         | **+1.3**          |

