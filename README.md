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
| 1     | ![Image 1](GeneratedImages/ApplesOranges/GeneratedApples/Original/1.jpg) | ![Image 2](GeneratedImages/ApplesOranges/GeneratedApples/CycleGAN/orange1.jpg) | ![Image 3](GeneratedImages/ApplesOranges/GeneratedApples/CycleMobiGAN/orange1.jpg) | ![Image 4](GeneratedImages/ApplesOranges/GeneratedApples/CycleMobiGANV2/orange1.jpg) |
| 2     | ![Image 5](GeneratedImages/ApplesOranges/GeneratedApples/Original/2.jpg) | ![Image 6](GeneratedImages/ApplesOranges/GeneratedApples/CycleGAN/orange2.jpg) | ![Image 7](GeneratedImages/ApplesOranges/GeneratedApples/CycleMobiGAN/orange2.jpg) | ![Image 8](GeneratedImages/ApplesOranges/GeneratedApples/CycleMobiGANV2/orange2.jpg) |
| 3     | ![Image 9](GeneratedImages/ApplesOranges/GeneratedApples/Original/3.jpg) | ![Image 10](GeneratedImages/ApplesOranges/GeneratedApples/CycleGAN/orange3.jpg) | ![Image 11](GeneratedImages/ApplesOranges/GeneratedApples/CycleMobiGAN/orange3.jpg) | ![Image 12](GeneratedImages/ApplesOranges/GeneratedApples/CycleMobiGANV2/orange3.jpg) |

Figure 2: Generated images in domain X (apples) using images in domain Y (oranges). Y -> X mapping
|       | Real Image                    | CycleGAN                      | Cycle-MobiGAN V1              | Cycle-MobiGAN V2              |
|-------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|
| 1     | ![Image 1](GeneratedImages/ApplesOranges/GeneratedOranges/Original/1.jpg) | ![Image 2](GeneratedImages/ApplesOranges/GeneratedOranges/CycleGAN/apple1.jpg) | ![Image 3](GeneratedImages/ApplesOranges/GeneratedOranges/CycleMobiGAN/apple1.jpg) | ![Image 4](GeneratedImages/ApplesOranges/GeneratedOranges/CycleMobiGANV2/apple1.jpg) |
| 2     | ![Image 5](GeneratedImages/ApplesOranges/GeneratedOranges/Original/2.jpg) | ![Image 6](GeneratedImages/ApplesOranges/GeneratedOranges/CycleGAN/apple2.jpg) | ![Image 7](GeneratedImages/ApplesOranges/GeneratedOranges/CycleMobiGAN/apple2.jpg) | ![Image 8](GeneratedImages/ApplesOranges/GeneratedOranges/CycleMobiGANV2/apple2.jpg) |
| 3     | ![Image 9](GeneratedImages/ApplesOranges/GeneratedOranges/Original/3.jpg) | ![Image 10](GeneratedImages/ApplesOranges/GeneratedOranges/CycleGAN/apple3.jpg) | ![Image 11](GeneratedImages/ApplesOranges/GeneratedOranges/CycleMobiGAN/apple3.jpg) | ![Image 12](GeneratedImages/ApplesOranges/GeneratedOranges/CycleMobiGANV2/apple3.jpg) |

