<img src="./images/denoising-diffusion.png" width="500px"></img>

## Denoising Diffusion Probabilistic Model, in Pytorch

Implementation of <a href="https://arxiv.org/abs/2006.11239">Denoising Diffusion Probabilistic Model</a> in Pytorch. It is a new approach to generative modeling that may <a href="https://ajolicoeur.wordpress.com/the-new-contender-to-gans-score-matching-with-langevin-sampling/">have the potential</a> to rival GANs. It uses denoising score matching to estimate the gradient of the data distribution, followed by Langevin sampling to sample from the true distribution.

This implementation was inspired by the official Tensorflow version <a href="https://github.com/hojonathanho/diffusion">here</a>

Youtube AI Educators - <a href="https://www.youtube.com/watch?v=W-O7AZNzbzQ">Yannic Kilcher</a> | <a href="https://www.youtube.com/watch?v=344w5h24-h8">AI Coffeebreak with Letitia</a> | <a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">Outlier</a>

<a href="https://github.com/yiyixuxu/denoising-diffusion-flax">Flax implementation</a> from <a href="https://github.com/yiyixuxu">YiYi Xu</a>

<a href="https://huggingface.co/blog/annotated-diffusion">Annotated code</a> by Research Scientists / Engineers from <a href="https://huggingface.co/">🤗 Huggingface</a>

Update: Turns out none of the technicalities really matters at all | <a href="https://arxiv.org/abs/2208.09392">"Cold Diffusion" paper</a> | <a href="https://muse-model.github.io/">Muse</a>

<img src="./images/sample.png" width="500px"><img>

[![PyPI version](https://badge.fury.io/py/denoising-diffusion-pytorch.svg)](https://badge.fury.io/py/denoising-diffusion-pytorch)

## Install

```bash
$ pip install denoising_diffusion_pytorch
```

## Usage

```python
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000    # number of steps
)

training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()

# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)
```

Or, if you simply want to pass in a folder name and the desired image dimensions, you can use the `Trainer` class to easily train a model.

```python
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    'path/to/your/images',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()
```

Samples and model checkpoints will be logged to `./results` periodically

## Multi-GPU Training

The `Trainer` class is now equipped with <a href="https://huggingface.co/docs/accelerate/accelerator">🤗 Accelerator</a>. You can easily do multi-gpu training in two steps using their `accelerate` CLI

At the project root directory, where the training script is, run

```python
$ accelerate config
```

Then, in the same directory

```python
$ accelerate launch train.py
```

## Miscellaneous

### 1D Sequence

By popular request, a 1D Unet + Gaussian Diffusion implementation.

```python
import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 32
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 128,
    timesteps = 1000,
    objective = 'pred_v'
)

training_seq = torch.rand(64, 32, 128) # features are normalized from 0 to 1

loss = diffusion(training_seq)
loss.backward()

# Or using trainer

dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()

# after a lot of training

sampled_seq = diffusion.sample(batch_size = 4)
sampled_seq.shape # (4, 32, 128)

```

`Trainer1D` does not evaluate the generated samples in any way since the type of data is not known.

You could consider adding a suitable metric to the training loop yourself after doing an editable install of this package
`pip install -e .`.

## Citations

```bibtex
@inproceedings{NEURIPS2020_4c5bcfec,
    author      = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
    pages       = {6840--6851},
    publisher   = {Curran Associates, Inc.},
    title       = {Denoising Diffusion Probabilistic Models},
    url         = {https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf},
    volume      = {33},
    year        = {2020}
}
```

```bibtex
@InProceedings{pmlr-v139-nichol21a,
    title       = {Improved Denoising Diffusion Probabilistic Models},
    author      = {Nichol, Alexander Quinn and Dhariwal, Prafulla},
    booktitle   = {Proceedings of the 38th International Conference on Machine Learning},
    pages       = {8162--8171},
    year        = {2021},
    editor      = {Meila, Marina and Zhang, Tong},
    volume      = {139},
    series      = {Proceedings of Machine Learning Research},
    month       = {18--24 Jul},
    publisher   = {PMLR},
    pdf         = {http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf},
    url         = {https://proceedings.mlr.press/v139/nichol21a.html},
}
```

```bibtex
@inproceedings{kingma2021on,
    title       = {On Density Estimation with Diffusion Models},
    author      = {Diederik P Kingma and Tim Salimans and Ben Poole and Jonathan Ho},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
    year        = {2021},
    url         = {https://openreview.net/forum?id=2LdBqxc1Yv}
}
```

```bibtex
@article{Karras2022ElucidatingTD,
    title   = {Elucidating the Design Space of Diffusion-Based Generative Models},
    author  = {Tero Karras and Miika Aittala and Timo Aila and Samuli Laine},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2206.00364}
}
```

```bibtex
@article{Song2021DenoisingDI,
    title   = {Denoising Diffusion Implicit Models},
    author  = {Jiaming Song and Chenlin Meng and Stefano Ermon},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2010.02502}
}
```

```bibtex
@misc{chen2022analog,
    title   = {Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning},
    author  = {Ting Chen and Ruixiang Zhang and Geoffrey Hinton},
    year    = {2022},
    eprint  = {2208.04202},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@article{Salimans2022ProgressiveDF,
    title   = {Progressive Distillation for Fast Sampling of Diffusion Models},
    author  = {Tim Salimans and Jonathan Ho},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2202.00512}
}
```

```bibtex
@article{Ho2022ClassifierFreeDG,
    title   = {Classifier-Free Diffusion Guidance},
    author  = {Jonathan Ho},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2207.12598}
}
```

```bibtex
@article{Sunkara2022NoMS,
    title   = {No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects},
    author  = {Raja Sunkara and Tie Luo},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.03641}
}
```

```bibtex
@inproceedings{Jabri2022ScalableAC,
    title   = {Scalable Adaptive Computation for Iterative Generation},
    author  = {A. Jabri and David J. Fleet and Ting Chen},
    year    = {2022}
}
```

```bibtex
@article{Cheng2022DPMSolverPlusPlus,
    title   = {DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models},
    author  = {Cheng Lu and Yuhao Zhou and Fan Bao and Jianfei Chen and Chongxuan Li and Jun Zhu},
    journal = {NeuRips 2022 Oral},
    year    = {2022},
    volume  = {abs/2211.01095}
}
```

```bibtex
@inproceedings{Hoogeboom2023simpleDE,
    title   = {simple diffusion: End-to-end diffusion for high resolution images},
    author  = {Emiel Hoogeboom and Jonathan Heek and Tim Salimans},
    year    = {2023}
}
```

```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.01327,
    doi     = {10.48550/ARXIV.2302.01327},
    url     = {https://arxiv.org/abs/2302.01327},
    author  = {Kumar, Manoj and Dehghani, Mostafa and Houlsby, Neil},
    title   = {Dual PatchNorm},
    publisher = {arXiv},
    year    = {2023},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

```bibtex
@inproceedings{Hang2023EfficientDT,
    title   = {Efficient Diffusion Training via Min-SNR Weighting Strategy},
    author  = {Tiankai Hang and Shuyang Gu and Chen Li and Jianmin Bao and Dong Chen and Han Hu and Xin Geng and Baining Guo},
    year    = {2023}
}
```

```bibtex
@misc{Guttenberg2023,
    author  = {Nicholas Guttenberg},
    url     = {https://www.crosslabs.org/blog/diffusion-with-offset-noise}
}
```

```bibtex
@inproceedings{Lin2023CommonDN,
    title   = {Common Diffusion Noise Schedules and Sample Steps are Flawed},
    author  = {Shanchuan Lin and Bingchen Liu and Jiashi Li and Xiao Yang},
    year    = {2023}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```

```bibtex
@article{Bondarenko2023QuantizableTR,
    title   = {Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing},
    author  = {Yelysei Bondarenko and Markus Nagel and Tijmen Blankevoort},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2306.12929},
    url     = {https://api.semanticscholar.org/CorpusID:259224568}
}
```

```bibtex
@article{Karras2023AnalyzingAI,
    title   = {Analyzing and Improving the Training Dynamics of Diffusion Models},
    author  = {Tero Karras and Miika Aittala and Jaakko Lehtinen and Janne Hellsten and Timo Aila and Samuli Laine},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2312.02696},
    url     = {https://api.semanticscholar.org/CorpusID:265659032}
}
```

```bibtex
@article{Li2024ImmiscibleDA,
    title   = {Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment},
    author  = {Yiheng Li and Heyang Jiang and Akio Kodaira and Masayoshi Tomizuka and Kurt Keutzer and Chenfeng Xu},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2406.12303},
    url     = {https://api.semanticscholar.org/CorpusID:270562607}
}
```

```bibtex
@article{Chung2024CFGMC,
    title   = {CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models},
    author  = {Hyungjin Chung and Jeongsol Kim and Geon Yeong Park and Hyelin Nam and Jong Chul Ye},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2406.08070},
    url     = {https://api.semanticscholar.org/CorpusID:270391454}
}
```

```bibtex
@inproceedings{Sadat2024EliminatingOA,
    title   = {Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models},
    author  = {Seyedmorteza Sadat and Otmar Hilliges and Romann M. Weber},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273098845}
}
```

----------------

# Repo Structure

## Quick plan
I'll summarize each file under `denoising_diffusion_pytorch/` based on the listing you provided and common diffusion repo patterns, and point out where downloads or cache logic typically live and what to open next.

## Files and what they likely contain

- `__init__.py`
  - Package initializer. Probably exposes main classes or convenience imports (e.g., from .denoising_diffusion_pytorch import Unet, GaussianDiffusion, etc.). Good place to see the public API.

- `attend.py`
  - Attention layers or modules used by the U-Net. Likely implements multi-head attention, spatial/channel attention, or cross-attention helpers used in the model.

- `classifier_free_guidance.py`
  - Utilities to run classifier-free guidance at sampling time. Likely contains functions that combine conditional and unconditional model outputs, or helper wrappers to run guidance.

- `continuous_time_gaussian_diffusion.py`
  - Implements diffusion processes parameterized in continuous time (t in [0,1]) rather than discrete timesteps. Contains schedulers, noise schedules, and probably sampling/denoising loops specialized to continuous-time parameterizations.

- `denoising_diffusion_pytorch_1d.py`
  - A 1D variant of the diffusion model (for audio or time-series). Contains a U-Net or diffusion training/sampling logic adapted to 1D convolutions.

- `denoising_diffusion_pytorch.py`
  - The main entrypoint for the discrete-time diffusion model. Expect:
    - Model class (U-Net) or importing/constructing it.
    - GaussianDiffusion trainer class: forward denoising, loss calculation, sampling.
    - Training loop helpers, sampling functions (p_sample_loop, sample, etc.).
  - This is the file to open first to understand training and sampling flow.

- `elucidated_diffusion.py`
  - Implementation of Elucidated Diffusion Models (EDM) — sampling and training modifications proposed in the EDM paper (Karras et al. style samplers, different noise scaling). Contains sampling code and specialized noise/scheduler utilities.

- `fid_evaluation.py`
  - Utilities to compute FID score (Fréchet Inception Distance) between generated images and a dataset. Likely wraps an Inception model, dataset loading, and FID calculation.

- `guided_diffusion.py`
  - Implements guidance strategies (e.g., classifier guidance or other conditioning mechanisms). Might include classifier-based guidance helpers and wrappers to apply gradient-based guidance during sampling.

- `karras_unet_1d.py`
  - A U-Net variant implementing Karras-style modifications for 1D signals (like the Karras et al. DDPM/EDM architectures). Contains building blocks and model class.

- `karras_unet_3d.py`
  - 3D U-Net variant for volumetric data (3D convolutions) with Karras-style enhancements.

- `karras_unet.py`
  - 2D U-Net with Karras et al. architecture tweaks (scaled activations, advanced resblocks, attention, residuals). This file often contains the model architecture used by EDM implementations.

- `learned_gaussian_diffusion.py`
  - Implements Gaussian diffusion where some parameters are learned rather than fixed (e.g., learned variance, parameterized noise schedules). Contains training and sampling code adapted to learned parameters.

- `repaint.py`
  - RePaint algorithm implementation — iterative masking/inpainting sampling algorithm for conditional inpainting tasks. Contains specific sampling loops that alternate masking and denoising.

- `simple_diffusion.py`
  - A minimal/simple diffusion implementation for clarity or debugging. Good reference to understand core ideas without bells and whistles.

- `v_param_continuous_time_gaussian_diffusion.py`
  - Continuous-time diffusion variant that parameterizes the model output as v (a mix of noise and denoised sample), following the v-parameterization used in some papers. Contains scheduler and training/sampling loops.

- `version.py`
  - Package version string (e.g., __version__ variable). Used by setup.py or to check package version.

- `weighted_objective_gaussian_diffusion.py`
  - Gaussian diffusion training with weighted objectives (e.g., per-timestep weighting, reweighting losses). Contains the loss computation logic and possibly schedules for weights.

- `classifier_free_guidance.py` (note: already listed above — if there are two similarly named files, one might be duplicated; check file contents)
  - (If present twice, one might be the public wrapper and another a more experimental version — open both to confirm.)

## Where downloads or large-file caching usually appear
This repo appears to be mostly model and training code (no obvious HF hub code in the filenames). Places downloads or large-file cache logic would typically appear:
- A `scripts/` folder or top-level `demo.py`, `setup.py`, or README instructions (not in the `denoising_diffusion_pytorch/` folder you listed).
- `fid_evaluation.py` might download an Inception model checkpoint or use torchvision/huggingface utilities that cache weights.
- Any file that calls `torch.hub.load`, `from_pretrained`, or Hugging Face functions could trigger downloads. Search terms: `torch.hub`, `load_state_dict_from_url`, `model_zoo`, `hf_hub_download`, `from_pretrained`, `inception`, `download`.
- `karras_unet*` or `*unet.py` may include pretrained checkpoints in examples or tests, but not typical.

If you want me to find exact download lines, I can search the repo for the terms above and open the matching files.

## Recommended next steps
- If you want exact locations of downloads/cache code, tell me and I will:
  - Search for strings: `torch.hub.load`, `hf_hub_download`, `from_pretrained`, `load_state_dict_from_url`, `cached_path`, `model_zoo`, `inception`, `download`.
  - Open files that match and change default cache path logic (e.g., use XDG or an env var).
- If you want I can open `denoising_diffusion_pytorch.py` and `karras_unet.py` to give a short tour of the training and sampling API (methods and parameters).

What would you like next — search for download calls, or open specific files for a deeper walkthrough?