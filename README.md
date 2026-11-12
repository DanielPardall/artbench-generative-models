# ArtBench Generative Models

**A PyTorch-based academic study of artistic image generation with GANs, variational autoencoders and diffusion models.**

The project explores different ways to learn the distribution of ArtBench-10 images and generate new 32 × 32 RGB samples. It combines model development, hyperparameter experiments, image sampling and evaluation using Fréchet Inception Distance (FID) and Kernel Inception Distance (KID).

Developed by **Daniel Monteiro and Gustavo Sousa** for *Inteligência Artificial Generativa*, University of Coimbra.

> **Repository scope:** this release contains the three notebooks supplied with the project. Dataset files, the dataset loader, the VAE/diffusion support modules and trained checkpoints were not included in the source archive. The notebooks are therefore not independently runnable as provided.

## Models explored

| Family | Models | Available implementation |
| --- | --- | --- |
| Adversarial generation | DCGAN, WGAN-GP | Generator/discriminator or critic definitions, training functions, tuning and evaluation code within the notebooks |
| Autoencoders | β-VAE, conditional VAE, VQ-VAE, spatial β-VAE | Notebook calls to external model, training and sampling modules; those modules are not bundled |
| Diffusion | Pixel-space DDPM and latent diffusion | Notebook configuration, training, checkpoint loading and evaluation calls; implementation modules are not bundled |

The VQ-VAE workflow also references a Transformer-based prior. The latent diffusion workflow uses a spatial VAE to encode/decode images and a diffusion model in the latent representation.

## What the notebooks cover

- Loading ArtBench-10 training and test splits through a local dataset helper.
- Building a development subset from `training_20_percent.csv`.
- Image preprocessing, class-distribution inspection and sample grids.
- PyTorch model configuration and optimization.
- GAN hyperparameter experiments, validation-based early stopping and checkpoint management.
- Sampling and comparison of real and generated images.
- FID and KID evaluation, including code for repeated evaluation runs.
- VAE and diffusion experiments driven by an external project module tree.

The GAN notebooks include techniques such as PixelNorm and, for DCGAN, optional spectral normalization. WGAN-GP includes gradient-penalty computation. The diffusion notebook references exponential moving averages (EMA) during training and sampling.

## Dataset and preprocessing

The project uses **ArtBench-10**, with ten artistic-style classes, and trains on images resized to **32 × 32 pixels**.

The GAN notebooks expect a local helper returning dataset splits with `image` and `label` fields. They read subset indices from a CSV column named `train_id_original`, then split the selected subset into training and validation partitions. GAN image tensors are normalized to `[-1, 1]`; the VAE/diffusion notebook initially constructs tensors in `[0, 1]`, with further processing dependent on its external modules.

The dataset is not redistributed in this repository. Recover the original dataset layout and accompanying loader before trying to reproduce the experiments: an arbitrary folder of images is not a drop-in replacement for the loader's expected structure.

## Repository structure

| Path | Contents |
| --- | --- |
| [`notebooks/gans/DCGAN.ipynb`](notebooks/gans/DCGAN.ipynb) | DCGAN architectures, configuration, training, tuning, sampling and metrics |
| [`notebooks/gans/WGAN_GP.ipynb`](notebooks/gans/WGAN_GP.ipynb) | WGAN-GP architectures, gradient penalty, training and evaluation |
| [`notebooks/vaes-diffusion/VAEs_DDPM.ipynb`](notebooks/vaes-diffusion/VAEs_DDPM.ipynb) | VAE, DDPM and latent diffusion experiment orchestration |
| `.gitignore` | Excludes local environments, generated results, datasets and checkpoints |

Notebook outputs and execution counts have been removed to keep the repository compact. Code cells are preserved. The original report PDF is excluded from this code-focused package.

## Dependencies and original environment

The notebooks were written for **Google Colab with Google Drive storage** and use PyTorch. A CUDA-capable GPU is recommended for training; several VAE/diffusion cells explicitly request `cuda` and require changes for CPU execution.

Visible imports use:

- `torch`, `torchvision`
- `torchmetrics`, `torch-fidelity`
- `numpy`, `pandas`, `matplotlib`
- `Pillow`, `tqdm`, `IPython`
- `datasets` for the dataset workflow

Jupyter is needed for local notebook use. The missing support modules may introduce additional dependencies. No original environment lockfile was supplied, so exact package-version compatibility has not been established.

## Recovering the runnable environment

### GAN notebooks

The original configuration uses:

| Item | Expected location relative to `ROOT` |
| --- | --- |
| Dataset | `archive/` |
| Subset index file | `archive/training_20_percent.csv` |
| Dataset helper | `src/artbench_local_dataset.py` |
| Generated artifacts | `results/models/`, `results/images/`, `results/metrics/` |

`ROOT` originally points to `/content/drive/MyDrive/IAG`.

To run these notebooks:

1. Restore the original dataset, index CSV and `artbench_local_dataset.py` helper.
2. Open the notebook in Colab and select the intended runtime, or prepare a compatible local PyTorch environment.
3. Adjust `ROOT` to your project location. For local execution, replace the Google Drive mount cell with local path configuration.
4. Review the experiment settings, output paths and any cells that load existing checkpoints before running them.
5. Execute the relevant sections in order. Start with small training and evaluation settings to verify the environment before launching full experiments.

### VAE and diffusion notebook

This notebook originally changes its working directory to `/content/drive/MyDrive/IAG_Project1`. It expects:

- `datasets/archive/` and `datasets/training_20_percent.csv`.
- `src/a_Data/`: dataset loading and sample export helpers.
- `src/b_Train/models/autoencoders/` and `src/b_Train/models/diffusion/`: model implementations.
- `src/b_Train/tune_train/`: training and hyperparameter-search modules.
- `src/c_Evaluation/`: sampling and metric evaluation utilities.
- `src/utils/plots.py`: plotting helpers.
- Trained checkpoints for cells that load saved `.pt` files.

**These directories and checkpoints are absent from this release.** Restore the original support files and update absolute paths before executing the notebook. Installing PyTorch alone will not resolve the missing project imports.

## Evaluation

The notebooks use **FID** and **KID** to compare real and generated image distributions, alongside qualitative sample inspection. They contain different development and final-evaluation settings; comparisons should use consistent sample counts, real-image splits and preprocessing.

No new experiments were run for this repository preparation. Numeric results are not reproduced here because the required data, support modules and checkpoints are not bundled. This repository should be presented as academic experiment code, not a pretrained image-generation service or an independently verified benchmark.

## Validation performed for this package

- Confirmed that each notebook remains valid notebook-format JSON.
- Verified that code and markdown cell sources match the supplied originals.
- Removed saved outputs and execution counts.
- Checked relative README links and archive integrity.

Training, sampling and runtime imports were not executed.

## Authors

- Daniel Monteiro
- Gustavo Sousa

Collaborative academic project — Generative Artificial Intelligence, University of Coimbra.
