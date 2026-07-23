# SplitterNet and other Efficient Image Denoising Models

<p align="center">
  <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Flepp_Real-World_Mobile_Image_Denoising_Dataset_with_Efficient_Baselines_CVPR_2024_paper.pdf"><img src="https://img.shields.io/badge/CVPR-2024-blue.svg" alt="CVPR 2024"></a>
  <a href="https://aiff22.github.io/midd.html"><img src="https://img.shields.io/badge/Project-Page-green.svg" alt="Project Page"></a>
  <a href="https://download.ai-benchmark.com/s/Gq3n2cS7QkH7ZMz"><img src="https://img.shields.io/badge/Dataset-MIDD-orange.svg" alt="MIDD Dataset"></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="License"></a>
</p>

<p align="center">
  <img src="images/SplitterNet_arch.png" width="90%" alt="SplitterNet Architecture"/>
</p>

<p align="center">
  Official implementation of the CVPR 2024 paper:<br/>
  <b>"Real-World Mobile Image Denoising Dataset with Efficient Baselines"</b>
</p>

---

## 📌 Table of Contents
- [Overview](#-overview)
- [SplitterNet Architecture](#-splitternet-architecture)
- [Included Models](#-included-models)
- [Repository Structure](#-repository-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Usage](#-usage)
  - [1. Training](#1-training)
  - [2. Evaluation](#2-evaluation)
  - [3. Local Execution](#3-local-execution)
  - [4. TensorFlow Lite Conversion](#4-tensorflow-lite-conversion)
  - [5. SIDD Benchmark Submission](#5-sidd-benchmark-submission)
- [Citation](#-citation)
- [License](#-license)

---

## 📖 Overview

This repository contains the official TensorFlow and PyTorch implementations of **SplitterNet** and **MoDeNet**, presented in our CVPR 2024 paper. Additionally, it provides a comprehensive suite of SOTA efficient mobile denoising networks optimized for TensorFlow Lite deployment, alongside baseline implementations from the [MAI 2021 Challenge](https://arxiv.org/pdf/2105.08629v1.pdf).

The associated **Real-World Mobile Image Denoising Dataset (MIDD)** can be downloaded here:
👉 **[Download MIDD Dataset](https://download.ai-benchmark.com/s/Gq3n2cS7QkH7ZMz)**

<p align="center">
  <img src="images/SamsungS23Ultra_ISP_Comparison.png" width="80%" alt="Samsung S23 Ultra ISP Comparison"/>
</p>

<p align="center">
  <img src="images/Dataset_comparison.png" width="75%" alt="Dataset Comparison"/>
</p>

---

## 🏗 SplitterNet Architecture

**SplitterNet** is specifically engineered for high-performance, real-time image denoising on mobile devices. By strategically splitting tensor channels across stages and utilizing lightweight attention mechanisms (simple channel attention & spatial attention), it achieves superior trade-offs between PSNR/SSIM reconstruction quality and mobile execution latency on TensorFlow Lite.

---

## 🧩 Included Models

The repository provides modular, dynamic implementations in `models/`:

| Model | Description | Reference |
| :--- | :--- | :--- |
| **SplitterNet** | Proposed ultra-efficient mobile denoising architecture | Flepp et al. (CVPR 2024) |
| **MoDeNet** | Proposed dynamic multi-scale efficient denoiser | Flepp et al. (CVPR 2024) |
| **Dynamic_PlainNet** | Dynamic implementation of PlainNet architecture | [NAFNet Paper](https://arxiv.org/pdf/2204.04676v4.pdf) |
| **Dynamic_UNet_simple** | Lightweight dynamic U-Net baseline | Baseline |
| **Megvii** | Winner of MAI 2021 Real-Time Image Denoising Challenge | [MAI 2021](https://arxiv.org/pdf/2105.08629v1.pdf) |
| **NOAHTCV** | Runner-up of MAI 2021 Real-Time Image Denoising Challenge | [MAI 2021](https://arxiv.org/pdf/2105.08629v1.pdf) |
| **PlainNet** | Standard PlainNet implementation | NAFNet |

> **Note on Dynamic Models:** You can pass custom block configurations per U-Net stage (e.g. `[2, 2, 4, 8]`, `[2, 2, 2, 2]`) as well as customize filter counts.

---

## 📂 Repository Structure

```text
├── models/                  # TensorFlow model architecture definitions
├── model_weights/           # Pretrained SplitterNet checkpoint (.h5)
├── sidd_submission/         # SIDD sRGB Benchmark submission preparation script
├── data_preprocessing/      # Parallel patch extraction and preprocessing tools
├── pytorch/                 # PyTorch implementation variant with training & inference
├── scripts/                 # SLURM cluster training & evaluation launch scripts
├── images/                  # Figures and visual result comparisons
├── train.py                 # Main training script
├── evaluate.py              # Model evaluation & PSNR/SSIM metric calculation
├── converter.py             # TensorFlow Lite model conversion script
├── dataloader.py            # Custom tf.data pipeline & patch pairing loader
├── utils.py                 # Custom loss functions (Charbonnier, Edge, PSNR) & metrics
├── requirements.txt         # Package dependencies
└── README.md                # Project documentation
```

---

## ⚙️ Prerequisites & Installation

### Environment Setup
Clone the repository and install dependencies using Python 3.10+:

```bash
# Clone the repository
git clone https://github.com/rflepp/SplitterNet-Efficient-Mobile-Denoising-Models-CVPR2024.git
cd SplitterNet-Efficient-Mobile-Denoising-Models-CVPR2024

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Dataset Preparation
1. Download the [MIDD Dataset](https://download.ai-benchmark.com/s/Gq3n2cS7QkH7ZMz) or your target denoising dataset.
2. If uncropped, create image patches using `data_preprocessing/cropping_parallel.py`.
3. Format test sets into subfolders with `/original/` (noisy) and `/denoised/` (ground truth) images.

---

## 🚀 Usage

### 1. Training
To launch training on a GPU cluster (via SLURM):

```bash
./scripts/run_training.sh SplitterNet 20 16 [1,1,1,1] [1,1,1,1] path/to/train/patches/ path/to/test/images/ 5 path/to/pretrained/model
```
*Arguments:* `[Model Name]` `[Epochs]` `[Batch Size]` `[Enc Blocks]` `[Dec Blocks]` `[Train Path]` `[Test Path]` `[Filter Exponent (2^N)]` `[Pretrained Weights Path / None]`

### 2. Evaluation
To evaluate a trained model checkpoint on test data:

```bash
./scripts/run_evaluation.sh /path/to/model_checkpoint/ evaluate_saved_model /path/to/test/data/
```

### 3. Local Execution
For running locally on a desktop GPU or CPU:

```bash
python train.py SplitterNet 20 16 ./output [1,1,1,1] [1,1,1,1] path/to/train/patches/ path/to/test/images/ 5 None
```

### 4. TensorFlow Lite Conversion
Convert a trained model to `.tflite` for mobile benchmark deployment:

```bash
python converter.py
```

### 5. SIDD Benchmark Submission
To evaluate SplitterNet on the official [SIDD sRGB Benchmark](http://130.63.97.225/sidd/benchmark_submit.php):

```bash
python sidd_submission/prepare_submission_srgb_sidd.py
```
This automatically downloads `BenchmarkNoisyBlocksSrgb.mat` if not present locally, performs inference using `model_weights/SplitterNet_MIDD_model.h5`, and generates `SubmitSrgb.mat` ready for submission.

---

## 📝 Citation

If you find SplitterNet or our MIDD dataset useful in your research, please cite our paper:

```bibtex
@inproceedings{flepp2024real,
  title={Real-World Mobile Image Denoising Dataset with Efficient Baselines},
  author={Flepp, Roman and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={25774--25783},
  year={2024}
}
```

---

## 📄 License

Copyright (C) 2024 Roman Flepp. All rights reserved.  
Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).  
*Released for academic research use only.*
