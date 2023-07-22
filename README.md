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
| 1     | ![Image 1](Generated Images/ApplesOranges/Generated Apples/Original/1.jpg) | ![Image 2](Generated Images/ApplesOranges/Generated Apples/CycleGAN/apple1.jpg) | ![Image 3](Generated Images/ApplesOranges/Generated Apples/Cycle MobiGAN/apple1.jpg) | ![Image 4](Generated Images/ApplesOranges/Generated Apples/Cycle MobiGAN V2/apple1.jpg) |
| 2     | ![Image 5](Generated Images/ApplesOranges/Generated Apples/Original/2.jpg) | ![Image 6](Generated Images/ApplesOranges/Generated Apples/CycleGAN/apple2.jpg) | ![Image 7](Generated Images/ApplesOranges/Generated Apples/Cycle MobiGAN/apple2.jpg) | ![Image 8](Generated Images/ApplesOranges/Generated Apples/Cycle MobiGAN V2/apple2.jpg) |
| 3     | ![Image 9](Generated Images/ApplesOranges/Generated Apples/Original/3.jpg) | ![Image 10](Generated Images/ApplesOranges/Generated Apples/CycleGAN/apple3.jpg) | ![Image 11](Generated Images/ApplesOranges/Generated Apples/Cycle MobiGAN/apple3.jpg) | ![Image 12](Generated Images/ApplesOranges/Generated Apples/Cycle MobiGAN V2/apple3.jpg) |

Figure 2: Generated images in domain X (apples) using images in domain Y (oranges). Y -> X mapping
|       | Real Image                    | CycleGAN                      | Cycle-MobiGAN V1              | Cycle-MobiGAN V2              |
|-------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|
| 1     | ![Image 1](Generated Images/ApplesOranges/Generated Oranges/Original/1.jpg) | ![Image 2](Generated Images/ApplesOranges/Generated Oranges/CycleGAN/orange1.jpg) | ![Image 3](Generated Images/ApplesOranges/Generated Oranges/Cycle MobiGAN/orange1.jpg) | ![Image 4](Generated Images/ApplesOranges/Generated Oranges/Cycle MobiGAN V2/orange1.jpg) |
| 2     | ![Image 5](Generated Images/ApplesOranges/Generated Oranges/Original/2.jpg) | ![Image 6](Generated Images/ApplesOranges/Generated Oranges/CycleGAN/orange2.jpg) | ![Image 7](Generated Images/ApplesOranges/Generated Oranges/Cycle MobiGAN/orange2.jpg) | ![Image 8](Generated Images/ApplesOranges/Generated Oranges/Cycle MobiGAN V2/orange2.jpg) |
| 3     | ![Image 9](Generated Images/ApplesOranges/Generated Oranges/Original/3.jpg) | ![Image 10](Generated Images/ApplesOranges/Generated Oranges/CycleGAN/orange3.jpg) | ![Image 11](Generated Images/ApplesOranges/Generated Oranges/Cycle MobiGAN/orange3.jpg) | ![Image 12](Generated Images/ApplesOranges/Generated Oranges/Cycle MobiGAN V2/orange3.jpg) |

