# EEG-based Driver Drowsiness Classification via Calibration-Free Framework with Domain Generalization. (SMC 2022, Oral)

This is an official repo for EEG-based Driver Drowsiness Classification via Calibration-Free Framework with Domain Generalization. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9945216)

## Description

We propose an EEG-based driver drowsiness state (i.e., alert and drowsy) classification framework for calibration-free BCI from the domain generalization perspective. We gathered samples from  all the subjects/domains in a domain-balaced and class-balanced manner an composed a mini-batch. We combined the style features of multi-domain instances for data augmentation to generate unseen domains. Moreover, we aligned the class relationship by minimizing the distance of label distribution within classes. The distance between the center of soft labels and each soft labels within each class is minimized.

![](docs/overview.png)

## Getting Started

### Environment Requirement

Clone the repo:

```bash
git clone https://github.com/KDongYoung/EEG-based-Driver-Drowsiness-Classification-with-Domain-Generalization.git
```

Install the HANet requirements using `conda`:

```terminal
conda create -n MixAlign python=3.8.13
conda activate MixAlign
pip install -r requirements.txt
```

IF using a Docker, use the recent image file ("pytorch:22.04-py3") uploaded in the [\[NVIDIA pytorch\]](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) when running a container


## Data preparation

First, create a folder `${DATASET_DIR}` to store the data of each subject.

The directory structure should look like this:

```
${DATASET_DIR}
	|--${S1}
        |--${S2}
	|--${...}
```

We will prepare the dataset as soon as possible.

### Training from scratch

```shell script
# train
python TotalMain.py --mode train
# test
python TotalMain.py --mode infer
```

The (BEST model for each SUBJECT and the tensorboard records) are saved in `${MODEL_SAVE_DIR}/{seed}_{step}_{alignment loss weight}/{model_name}` by default

The results are saved in text and csv files in `${MODEL_SAVE_DIR}/{seed}_{step}_{alignment loss weight}/{Results}/{evalauation metric}` by default

-> The BEST models are saved separately in each folder based on the evaluation metric used to select the model for validation.

The result directory structure would look like this:

```
${MODEL_SAVE_DIR}
    ${seed}_{step}_{alignment loss weight}
	|--${model_name}
	    |--${models}
	    	|--${evaluation metric}
	    |--${tensorboard records}
        |--${Results}
	    |--${evaluation metric}
	    	|--${csv file}
		|--${txt file}
```

### Evaluation

**Results for drowsiness classification:**
| Model   | Acc. ± Std. (%)  | F1-score ± Std. | Recall ± Std. | Inference Time | Checkpoint |
| ------- | ---------------- | --------------- | ------------- | -------------- | ---------- |
| OURS    | 77.26 ± 10.44    | 0.6266 ± 0.24   | 0.6813 ± 0.25 | 3.853 ms       |  |

Acc.:  Accuracy,  Std.:  Standard  deviation


## Citation

```
@inproceedings{kim2022eeg,
  title={EEG-based Driver Drowsiness Classification via Calibration-Free Framework with Domain Generalization},
  author={Kim, Dong-Young and Han, Dong-Kyun and Jeong, Ji-Hoon and Lee, Seong-Whan},
  booktitle={2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  pages={2293--2298},
  year={2022},
  organization={IEEE}
}
```
