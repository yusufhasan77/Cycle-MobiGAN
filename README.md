# Cycle-MobiGAN

## Overview

This project was done as part of the bachelor thesis at Tilburg University. This repository contains more structured code with notebooks and results with different datasets. The purpose of this project was to investigate that to what extent does modifying CycleGAN architecture with depthwise separable convolution and inverted residual blocks as introduced in MobileNet and MobileNet v2 effect computational efficiency and generated image quality.
The methodology section below provides details of architectures and training parameters, whereas, the results section displays the results.

The structure of the repository is described below:

```bash
Cycle-MobiGAN/
├── Dataset/
│   ├── Dataset.py
├── Models/
│   ├── Blocks.py
│   └── Discriminators.py
│   ├── Generators.py
│   └── Loss_functions.py
├── Train py files/
│   ├── train_CycleGAN.py
│   └── train_CycleGAN_DWS.py
│   ├── train_CycleGAN_IR_DWS.py
└── Utils/
   ├── visualize_images.py
```

Each subfolder has it's own README.md that explains what files that folders contains and their purpose.

## Introduction

## Methodology

## Results


Figure 1: Generated images in domain Y (oranges) using images in domain X (apples). X -> Y mapping
|       | Real Image                    | CycleGAN                      | Cycle-MobiGAN V1              | Cycle-MobiGAN V2              |
|-------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|
| 1     | ![Image 1](images/image1.png) | ![Image 2](images/image2.png) | ![Image 3](images/image3.png) | ![Image 4](images/image4.png) |
| 2     | ![Image 5](images/image5.png) | ![Image 6](images/image6.png) | ![Image 7](images/image7.png) | ![Image 8](images/image8.png) |
| 3     | ![Image 9](images/image9.png) | ![Image 10](images/image10.png)| ![Image 11](images/image11.png)| ![Image 12](images/image12.png)|
| 4     | ![Image 13](images/image13.png)| ![Image 14](images/image14.png)| ![Image 15](images/image15.png)| ![Image 16](images/image16.png)|
| 5     | ![Image 17](images/image17.png)| ![Image 18](images/image18.png)| ![Image 19](images/image19.png)| ![Image 20](images/image20.png)|

Figure 2: Generated images in domain X (apples) using images in domain Y (oranges). Y -> X mapping
|       | Real Image                    | CycleGAN                      | Cycle-MobiGAN V1              | Cycle-MobiGAN V2              |
|-------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|
| 1     | ![Image 1](images/image1.png) | ![Image 2](images/image2.png) | ![Image 3](images/image3.png) | ![Image 4](images/image4.png) |
| 2     | ![Image 5](images/image5.png) | ![Image 6](images/image6.png) | ![Image 7](images/image7.png) | ![Image 8](images/image8.png) |
| 3     | ![Image 9](images/image9.png) | ![Image 10](images/image10.png)| ![Image 11](images/image11.png)| ![Image 12](images/image12.png)|
| 4     | ![Image 13](images/image13.png)| ![Image 14](images/image14.png)| ![Image 15](images/image15.png)| ![Image 16](images/image16.png)|
| 5     | ![Image 17](images/image17.png)| ![Image 18](images/image18.png)| ![Image 19](images/image19.png)| ![Image 20](images/image20.png)|

