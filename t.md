# Homework 5 (Lensless Computational Imaging)

## Task

In this homework, you will get familiar with Lensless Computational Imaging and reproduce some algorithms and experiments from the papers.

> [!IMPORTANT]
> While there are plenty of open-source implementations, including those linked in the papers, **you cannot use\look at them**. You need to implement everything yourself from scratch. Similarly to the previous homework, you will find all required information here in the README, in the papers, their appendixes, or references. If something is missing, think of a reasonable design choice yourself.

Plan:

1. Read [Towards Robust and Generalizable Lensless Imaging with Modular Learned Reconstruction](https://arxiv.org/abs/2502.01102) to get familiar with core concepts.

2. Implement Unrolled ADMM (Le-ADMM) algorithm using [Learned reconstructions for practical mask-based lensless imaging](https://arxiv.org/abs/1908.11502). **Note:** if you fix hyperparameters in Unrolled ADMM, you get standard ADMM. (Hyperparameters: $\mu_i = 10^{-4}, \tau_i=2\cdot 10^{-4}$).

**Hints:**:

- Use Fourier for efficient computations.
- Use anisotropic version of TV regularizer instead of vectorial from the paper.
- For finite-differences (for TV), use a circular version to preserve shape.
- For handling the crop operation, consider working on roughly 2Hx2W _padded_ space with the actual image at the center, and crop the image at the end after all ADMM iterations. For faster FFT, you can pad even more to work at a more FFT-friendly space size.
- Use all-zeros as the initial estimate.

3. Using these papers, implement and compare:

- ADMM-100 (100 iterations) with fixed hyperparameters.
- Unrolled ADMM-20 (trainable).
- 8M-modular LeADMM-5 variants from the paper with (i) pre- and post-processors, (ii) only pre-processor, (iii) only post-processor. No need for the PSF-correction version.

We provide some helper functions in `lensless_helpers` folder and [Lensless notebook](./Lensless.ipynb).

Metrics: PSNR, LPIPS (VGG-variant), MSE, SSIM.

Dataset:

- Training: the **train** split of [DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K)
- Evaluation: the **test** split of [DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K)

For speed up, we limit quality metrics in [Grade](#grade) section to the results obtained only after about 6 hours of training (on Kaggle).

## Code requirements

Your submission **is the GitHub repository**; the code **is written in the project-style, not `ipynb` notebooks**.

> [!IMPORTANT]
> You **must** use the [Project Template](https://github.com/Blinorot/pytorch_project_template) and modify it. Run the code via scripts. Feel free to choose any of the code branches as a starting point. You must write clean code as in the previous homeworks. You will be penalized for unnecessary copy-paste and unreadable code.

Besides, you need to provide demo (see [Demo](#demo)). Hence, you need to save your final model checkpoints and provide them with your submission. You can use HuggingFace Models or Google Drive.

**Hint**: to run project style code in Colab\Kaggle, you can do the following:

```bash
!git clone https://YOUR_PRIVATE_TOKEN@github.com/USERNAME/REPO_ID

# change the root dir
%cd REPO_ID

# install packages
...

# run the code
python3 train.py ...
```

Code configuration and `%%writefile` will help you to avoid unnecessary commits.

## Other requirements

We do not accept the homework if any of the following requirements are not satisfied:

- The code shall be stored in a public github (or gitlab) repository. (Before the deadline, use a private repo. Make it public after the deadline.)
- All the necessary packages shall be mentioned in `./requirements.txt` or an analog (e.g. it can be `.lock` if you use `uv`).
- All necessary resources (e.g. model checkpoints) shall be downloadable with a script. Mention the script (or lines of code) in the `README.md`.
- `README.md` must clearly describe what the repo is about, how to install it and run the model training/inference.
- You must use `W&B`/`Comet ML` for logging losses, objects (like images), and performance metrics.
- You shall support dataset inference with an inference script and metrics calculation (see [Testing](#testing) section).
- You shall create a demo notebook (see [Demo](#demo) section).
- You must provide the logs for the training of **all** your final models from the start of the training. Use W&B (CometML) Reports feature.
- Attach a detailed report that includes:
  - Description and result of each experiment.
  - How to reproduce your model?
  - Attach training logs showing the rate of convergence and reconstructed images. Attach metrics curves.
  - What worked and what didn't work?
  - What were the major challenges?
  - Special tasks from the [Report](#report) section.

> [!TIPS]
> One of the homework goals is to get practice in proper project development. Thus, we will look at your Git history, `README`, `requirements.txt`, etc. Provide a comprehensive `README`, do not indicate packages that are not actually needed in the requirements, write meaningful commit names, etc. Do not hesitate to use `pre-commit` for code formatting (template version includes `black` and `isort`).

## Testing

You **must** add `inference.py` script and a `CustomDirDataset` Dataset class in `src/datasets/` with a proper config in `src/configs/`.

The `CustomDirDataset` shall be able to parse any directory of the following format:

```bash
NameOfTheDirectoryWithData
├── lensless
│   ├── ImageID1.png
│   ├── ImageID2.png
│   .
│   .
│   .
│   └── ImageIDn.png
├── masks
│   ├── ImageID1.npy
│   ├── ImageID2.npy
│   .
│   .
│   .
│   └── ImageIDn.npy
└── lensed # ground truth original image, may not exist
    ├── ImageID1.png
    ├── ImageID2.png
    .
    .
    .
    └── ImageIDn.png
```

It shall has an argument for the path to this custom directory that can be changed via `Hydra`-options.

The `inference.py` script must apply the model on the given dataset (custom-one or any other supported in your `src`) and save reconstructions in the requested directory. The predicted reconstruction id should be the same as the image id (so they can be matched together).

**Provide a separate script** that calculates metrics given the path to ground truth and reconstructed images.

Mention the lines on how to run inference on your final model in the `README`. Include the lines for the metrics calculation script too.

## Demo

In addition to providing detailed instructions in the `README`, a great repository shows a demo of the project, i.e., showcases how to use it. This allows end-user to see directly how to apply your code for their needs. The basic version of demo is the inference notebook. You must include such an `ipynb` notebook in your repository.

The structure of the demo:

1. Clones your repository and follows all the installation steps from your `README`.
2. Downloads all the required checkpoints.
3. Given a custom url (provided by a user) to a `.zip` dataset file on Google Drive, downloads it, processes it via `inference.py` and saves reconstructions at a user-defined path (mention the default one).
4. Visualizes some samples: original image (**if exists in the dataset**) vs lensless measurement vs reconstruction.
5. Given path to the unzipped original data and saved reconstructions, runs `calculate_metrics.py` and prints the 4 metrics on the screen if the original images exist in the provided dataset.
6. All of this is accompanied with comments explaining what your cell is doing and/or gives some instructions.

> [!IMPORTANT]
> Be sure to check your demo yourself. If the demo is absent or not working, you will get $0$ for the homework. We will use your demo notebook with our dataset links to evaluate your submission. **A user only have to pass the link and run the cells to get reconstructed samples, no other steps shall be expected from the user.**

To ensure fair evaluation, use Google Colab as the testing server for the verification of your demo. If your demo code works in a fresh Colab session, then, there shall not be any problems. If it doesn't work in Colab, we will penalize it.

## Report

In your report, you must include the following information:

1. For each method, how is it different from the previous one, what are its benefits and what are its drawbacks?
2. Qualitative and Quantitative comparison of all methods with analysis.

For comparison, also calculate reconstruction speed for all methods (code for doing so must be provided in the repository as well).

## Grade

```
grade = 0.6 * Metrics + 0.4 * (quality of code and report) - penalties
```

`Metrics` is based on [PSNR](https://lightning.ai/docs/torchmetrics/stable/image/peak_signal_noise_ratio.html). Calculated using your top model.

| Grade | PSNR    |
| ----: | ------- |
|    10 | > 16.25 |
|     8 | > 15.50 |
|     6 | > 14.00 |
|     4 | > 13.00 |

We lowered the metrics so they are achievable in around 6 hours, though you can train more for better results if you want.

## Bonuses

- `1.0 point`: implement a non-ADMM-based algorithm and compare (qualitatively and quantitatively) it with your other algorithms. In your report, explain how the method is different and what are its pros and cons. **Again, you cannot use implementations from the web.** Algorithm options (choose one): [FISTA](https://www.ceremade.dauphine.fr/~carlier/FISTA), [Trainable Inversion](https://arxiv.org/pdf/2010.15440), [MWDNs](https://doi.org/10.1364/oe.501970).

- `0.5 point`: choose a proper **pre-trained** super-resolution \ image restoration network (GAN or Diffusion-based, **general-purpose, not specifically designed for lensless**) and run it on top of vanilla ADMM-100 (fixed, not unrolled). Compare (qualitatively and quantitatively) it with your other algorithms. In your report, explain how the method is different and what are its pros and cons. **Make sure this model code is also a part of your project-style repo, following the best practices**.
  - `extra 1.0 point`: fine-tune this GAN\Diffusion on a training split of the dataset and do the comparison and discussion again. **Training code must also be a part of your project**.