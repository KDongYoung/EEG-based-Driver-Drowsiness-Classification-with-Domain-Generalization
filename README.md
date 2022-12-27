# EEG-based Driver Drowsiness Classification via Calibration-Free Framework with Domain Generalization. (SMC 2022, Oral)

This is an official repo for EEG-based Driver Drowsiness Classification via Calibration-Free Framework with Domain Generalization. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9945216)

## Description

We propose an EEG-based driver drowsiness state (i.e., alert and drowsy) classification framework for calibration-free BCI from the domain generalization perspective. We gathered samples from  all the subjects/domains in a domain-balaced and class-balanced manner an composed a mini-batch. We combined the style features of multi-domain instances for data augmentation to generate unseen domains. Moreover, we aligned the class relationship by minimizing the distance of label distribution within classes. The distance between the center of soft labels and each soft labels within each class is minimized.

![](docs/overview.png)

## Getting Started

1. Environment Requirement.

```terminal
conda create -n MixAlign python=3.8.13
conda activate MixAlign
pip install -r requirements.txt
```
