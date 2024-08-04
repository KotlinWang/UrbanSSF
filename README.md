<<<<<<< HEAD
# Accurate Segmentation of Urban Spatial Structure: A Framework for Large-Scale Remote Sensing Images Using Feature State Sequences

https://github.com/user-attachments/assets/df3c2974-1717-4e07-b301-4f6274e5ecd8.mp4

## :newspaper:News

- **[2024/8/4]** UrbanSSF Project Creation :sunglasses:. 

## :star:Overview

![overview](./assets/urbanssf.jpg)
- UrbanSSF is the first to combine CNNs, Transformers and Mamba for the remote sensing of VHR urban scenes. The Global Semantic Enhancer (GSE) module and the spatial interactive attention (SIA) mechanism process different scale features from the encoder. FSI Mamba uses the powerful sequence modeling ability of state Space Module (SSMs) to apply to the feature state sequence. Channel Space Reconstruction (CSR) algorithm is designed to reduce the computational complexity of large-scale feature fusion.
- UrbanSSF has achieved the effect of SOTA on three urban scene datasets of UAVid, ISPRS Vaihingen and Potsdam. Especially on the UAVid dataset.

## :bookmark_tabs: Main Result

| **Type**                  | **Method**     | **UAVid mIoU↑** | **Vaihingen OA↑** | **Vaihingen mF1↑** | **Vaihingen mIoU↑** | **Potsdam OA↑** | **Potsdam mF1↑** | **Potsdam mIoU↑** |
| ------------------------- | -------------- | --------------- | ----------------- | ------------------ | ------------------- | --------------- | ---------------- | ----------------- |
| **CNNs**                  | BiSeNet        | 61.5            | 87.1              | 84.3               | 75.8                | 88.2            | 89.8             | 81.7              |
|                           | DANet          | 60.6            | 88.2              | 79.6               | 69.4                | 89.1            | 88.9             | 80.3              |
|                           | SwiftNet       | 61.1            | 90.2              | 88.3               | 79.6                | 89.3            | 91.0             | 83.8              |
|                           | ABCNet         | 63.8            | 90.7              | 89.5               | 81.3                | 90.7            | 91.9             | 85.2              |
| **Transformer**           | Segmenter   | 58.7            | 88.1              | 84.1               | 73.6                | 88.7-           | 89.285.4         | 80.775.0          |
|                           | BANet       | 64.6            | 90.5              | 89.6               | 81.4                | 91.0            | 92.5             | 86.3              |
|                 | BoTNet      | 63.2            | 88.0              | 84.8               | 74.3                | -               | -                | -                 |
|                 | UNetFormer  | 66.1            | 91.0              | 90.4               | 82.7                | 90.8            | 92.0             | 85.3              |
| **Mamba**                 | Mamba-UNet     | 57.3            | 92.6          | 89.7               | 81.6                | 88.9            | 90.1             | 82.3              |
|                           | Swin-UMamba    | 53.4            | 92.4              | 89.4               | 81.3                | 89.1            | 90.4             | 82.7              |
|                           | VM-UNet        | 55.7            | 92.3              | 88.3               | 79.6                | 88.2            | 89.3             | 80.9              |
| **Our** | UrbanSSF-T     | 65.7            | 93.1              | 90.7               | 83.3                | 90.9            | 92.0             | 85.4              |
|                           | UrbanSSF-S     | 69.8            | 93.3              | 91.4               | 84.5               | 91.7   | 92.9    | 86.9     |
|                           | UrbanSSF-L     | **71.0**        | **93.6**          | **91.7**           | **85.0**            | **92.2**  | **93.3**   | **87.6**    |
| **Improvement**           |                | **+4.9**        | **+1.0**          | **+1.3**           | **+2.3**            | **+0.9**        | **+0.8**         | **+1.3**          |
