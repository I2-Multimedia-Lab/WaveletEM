# WaveletEM: Frequency-Aware Dual-Stream Learning for Balanced Realism and Fidelity in Electron Microscopy Imaging

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20115404.svg)](https://doi.org/10.5281/zenodo.20115404)

<img width="951" height="542" alt="image" src="https://github.com/user-attachments/assets/4c98d4b9-3b47-4aa7-bb4c-621a6bf47209" />

## 📖 Algorithm Introduction

Electron Microscopy (EM) imaging faces a fundamental trade-off between resolution and acquisition speed. Existing learning-based methods often rely on a single-stream architecture, struggling to balance perceptual realism and quantitative fidelity.

**WaveletEM** is the first dual-stream framework that explicitly decouples frequency components using the **Discrete Wavelet Transform (DWT)** to address this conflict:

1.  **Frequency-Aware Decomposition**: Leverages DWT to decompose an image into a low-frequency approximation (structure) and high-frequency detail components.
2.  **Dual-Stream Collaborative Reconstruction**:
    *   **Low-Frequency Stream**: Employs a **Conditional Diffusion Model (LFCDM)** to generate globally coherent structures with high biological realism.
    *   **High-Frequency Stream**: Uses a **Transformer-Based Model (HFTBM)** to accurately restore fine-grained textures and details.
3.  **Advantage**: Effectively combines the perceptual strengths of generative models with the quantitative accuracy of regression models, achieving state-of-the-art results on the EMDiffuse dataset in terms of LPIPS and resolution ratio.

## 📂 Data Preparation

The dataset used in this study is publicly available.

*   **Dataset**: [Download from Zenodo](https://zenodo.org/records/10205819)
*   **Pre-trained Weights**: [Google Drive Link](https://drive.google.com/drive/folders/1VT6aUdUCYJnBGv5sYmp9F_arRoftEgcB?usp=drive_link)
    *   Please download the weights and place them in the `checkpoints/` directory.

## ⚙️ Environment Setup

This project is developed with Python 3.8.

```bash
# 1. Clone the repository
git clone https://github.com/xiaogaogao26/WaveletEM.git
cd WaveletEM

# 2. Create and activate a Conda environment (recommended)
conda create -n waveletem python=3.8
conda activate waveletem

# 3. Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

The project is configured by modifying the files in the config/ directory. No code changes are needed.

1. Training

```bash
python train.py
```

2. Testing


```bash
python test.py
```

Please configure the path to your model checkpoint and test dataset in the relevant config file under config/ before running.

## Results

<img width="936" height="497" alt="image" src="https://github.com/user-attachments/assets/4a3ac803-0be1-4813-8c0c-bfb544391216" />


<img width="742" height="698" alt="image" src="https://github.com/user-attachments/assets/711c156d-3c67-4655-befd-aba5724c5286" />



## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@article{WaveletEM2026,
  title     = {Frequency-Aware Dual-Stream Learning for Balanced Realism and Fidelity in Electron Microscopy Imaging},
  author    = {Gao, L. and Zhao, Z. and Gao, P. and Paul, M.},
  journal   = {The Visual Computer},
  year      = {2026},
  note      = {Code: \url{https://doi.org/10.5281/zenodo.20115404}}
}
```
