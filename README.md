# Improving-ASR-with-LLM-Description
--------------------------------------
[[Paper]]()
Accepted to INTERSPEECH 2024

## Table of Contents
1. [Abstract](#abstract)
2. [Overview of our method](#overview-of-our-method)
3. [Setup](#setup)
    - [Conda Environment](#conda-environment)
    - [Dependency](#dependency)
4. [Run](#run)
    - [OCW](#ocw)
    - [Earnings Call](#earnings-call)
5. [Results](#results)
    - [Base.en](#baseen)
    - [Medium.en](#mediumen)

## Abstract
End-to-end automatic speech recognition (E2E ASR) systems have significantly improved speech recognition through training on extensive datasets. Despite these advancements, they still struggle to accurately recognize domain specific words, such as proper nouns and technical terminologies. To address this problem, we propose a method to utilize the state-of-the-art Whisper without modifying its architecture, preserving its generalization performance while enabling it to leverage descriptions effectively. Moreover, we propose two additional training techniques to improve the domain specific ASR: decoder fine-tuning, and context perturbation. We also propose a method to use a Large Language Model (LLM) to generate descriptions with simple metadata, when descriptions are unavailable. Our experiments demonstrate that proposed methods notably enhance domain-specific ASR accuracy on real-life datasets, with LLM-generated descriptions outperforming human-crafted ones in effectiveness.

### Overview of our method.
![Model Structure](./images/model_architecture.jpg)

### Setup
##### Conda Environment
```
conda create -n llm-description python=3.9 -y
conda activate llm-description
```
##### Dependency
```
sudo apt update && sudo apt install ffmpeg
pip install -r requirements.txt
```
### Run
Set dataset path(data_root) and save path(root_path) in whisper_fine.py
```
>>> data_root = "/data/jwsuh/whisper-datasets/main"
```
##### OCW
```
CUDA_VISIBLE_DEVICES=0 python whisper_fine.py  --dataset ocw --batch 32 --freeze
```
##### Earnings Call
```
CUDA_VISIBLE_DEVICES=0 python whisper_fine.py  --dataset earning --batch 32 --freeze
```

### Results
##### Base.en
| Models                     | Earnings Call (20 h) | Earnings Call (40 h) | OCW (20 h) | OCW (40 h) |
|----------------------------|----------------------|----------------------|------------|------------|
| Whisper (Frozen)           | 16.39%               | 16.39%               | 11.98%     | 11.98%     |
| + Full Fine-tuning         | 17.38%               | 16.64%               | 10.41%     | 9.94%      |
| + Description              | 20.63%               | 17.70%               | 9.81%      | 9.72%      |
| + Decoder Fine-tuning      | 16.61%               | 15.70%               | **9.79%**  | **9.67%**  |
| + Context Perturbation     | **16.24%**           | **15.15%**           | **9.79%**  | 9.68%      |

##### Medium.en
| Models                     | Earnings Call (20 h) | Earnings Call (40 h) | OCW (20 h) | OCW (40 h) |
|----------------------------|----------------------|----------------------|------------|------------|
| Whisper (Frozen)           | 16.39%               | 16.39%               | 11.98%     | 11.98%     |
| + Full Fine-tuning         | 17.38%               | 16.64%               | 10.41%     | 9.94%      |
| + Description              | 20.63%               | 17.70%               | 9.81%      | 9.72%      |
| + Decoder Fine-tuning      | 16.61%               | 15.70%               | **9.79%**  | **9.67%**  |
| + Context Perturbation     | **16.24%**           | **15.15%**           | **9.79%**  | 9.68%      |