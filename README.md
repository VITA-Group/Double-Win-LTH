# Data-Efficient Double-Win Lottery Tickets from Robust Pre-training

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Codes for this paper **Data-Efficient Double-Win Lottery Tickets from Robust Pre-training** [ICML 2022]

Tianlong Chen, Zhenyu Zhang, Sijia Liu, Yang Zhang, Shiyu Chang, Zhangyang Wang



## Overview

Pre-training serves as a broadly adopted starting point for transfer learning on various downstream tasks. Recent investigations of lottery tickets hypothesis (LTH) demonstrate such enormous pre-trained models can be replaced by extremely sparse subnetworks (a.k.a. matching subnetworks) without sacrificing transferability. 

However, practical security-crucial applications usually pose more challenging requirements beyond standard transfer, which also demand these subnetworks to overcome adversarial vulnerability. In this paper, we formulate a more rigorous concept, Double-Win Lottery Tickets, in which a located subnetwork from a pre-trained model can be independently transferred on diverse downstream tasks, to reach **BOTH** the same standard and robust generalization, under **BOTH** standard and adversarial training regimes, as the full pre-trained model can do. We comprehensively examine various pre-training mechanisms and find that robust pretraining tends to craft sparser double-win lottery tickets with superior performance over the standard counterparts. 

Furthermore, we observe the obtained double-win lottery tickets can be more data-efficient to transfer, under practical data-limited (e.g., 1% and 10%) downstream schemes. Our results show that the benefits from robust pre-training are amplified by the lottery ticket scheme, as well as the data-limited transfer setting.

## Prerequisites

```
pytorch == 1.5.1
torchvision == 0.6.1
advertorch == 0.2.3
```

## Usage

##### Iterative Magnitude Pruning (IMP) on pretraining tasks (ImageNet classification)

```
# IMP with adversarial training
bash script/imp_pretrain/imp_adv.sh [init-pretrained-weight] [save-direction] [data-direction]

# IMP with standard training
bash script/imp_pretrain/imp_std.sh [init-pretrained-weight] [save-direction] [data-direction]
```

##### Downstream training with located sparse subnetworks

```
# Adversarail training on CIFAR-10/100
bash script/train_downstream/adv_cifar.sh [dataset] [init-pretrained-weight] [save-direction] [located sparse structures]

# Adversarail training on SVHN
bash script/train_downstream/adv_svhn.sh [init-pretrained-weight] [save-direction] [located sparse structures]

# Standard training
bash script/std.sh [dataset] [init-pretrained-weight] [save-direction] [located sparse structures]
```

## Citation

```
TBD
```

