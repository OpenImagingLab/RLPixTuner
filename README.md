# Goal Conditioned Reinforcement Learning for Photo Finishing Tuning

[Project Page](https://openimaginglab.github.io/RLPixTuner/) | [Paper](https://openreview.net/pdf?id=4kVHI2uXRE/) | [Video](https://www.youtube.com/watch?v=fFIkc3KHS28)

This repo contains the code for [Goal Conditioned Reinforcement Learning for Photo Finishing Tuning](https://openreview.net/pdf?id=4kVHI2uXRE) (NeurIPS'24)

------

## Overview

RLPixTuner is an implementation of <br>
"[**Goal Conditioned Reinforcement Learning for Photo Finishing Tuning**](https://openimaginglab.github.io/RLPixTuner/)" <br>
Jiarui Wu, Yujin Wang, Lingen Li, Zhang Fan, Tianfan Xue <br>
in Conference on Neural Information Processing Systems (**NeurIPS**), 2024

![img](https://openimaginglab.github.io/RLPixTuner/static/images/teaser.png)

In this work, we propose an RL-based photo finishing tuning algorithm that efficiently tunes the parameters of a black-box photo finishing pipeline to match any tuning target. The RL-based solution (top row) takes only about 10 iterations to achieve a similar PSNR as the 500-iteration output of a zeroth-order algorithm (bottom row). Our method demonstrates fast convergence, high quality, and no need for a proxy.

## Quick Start

Tested with python 3.9.

### Environment

Clone our repo:

```
git clone https://github.com/OpenImagingLab/RLPixTuner.git
cd RLPixTuner
```

Install dependencies:

```
conda create --name <env> --file requirements.txt
```

### Data

Prepare the input-target data pairs like:

```
├─ExpertC0001-0-Best-Input.tif
└─ExpertC0001-0-Best-Target.tif
...
```

Here is the an example (subset) of the [FiveK-Random Evaluation Data](https://drive.google.com/file/d/1LgaaLnVm1MXrDTlFAMzka6_PbdqwR1yE/view?usp=sharing) we used in our paper.

After downloading the data, run:

```
python tools/process_data.py /path/to/your/dataset/directory
```

### Model

The pretrained weight for photo finishing tuning and photo stylization tuning comes with our repo:

```
envs/checkpoints/best_model.zip
envs/checkpoints/best_model_style.zip
```


## Usage

This repo currently contains implementation for photo finishing tuning and photo stylization tuning task in our paper.

### Inference

**Photo Finishing Tuning**

Specify the eval data path and model weight in `bash/run_pft.sh`. Then run:
```
bash bash/run_pft.sh
```

**Photo Stylization Tuning**

Specify the eval data path and model weight in `bash/run_pst.sh`. Then run:
```
bash bash/run_pst.sh
```

### Train

Specify the training data path in `bash/train.sh`. Then run:
```
bash bash/train.sh
```
We recommend tuning these hyper-parameters first: `agent_lr, value_lr, replay_size, ddpg_gamma, ema_rate`.

## Citation

If you use our work in your research, please use the following BibTeX entry.

```
@article{wu2024goal,
  title={Goal Conditioned Reinforcement Learning for Photo Finishing Tuning},
  author={Jiarui Wu and Yujin Wang and Lingen Li and Zhang Fan and Tianfan Xue},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
}
```

## License

<a rel="license" href="https://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://licensebuttons.net/l/by-nc/4.0/88x31.png" />

This repository is licensed under [Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/deed.en)
1. **Attribution** — Proper credit must be given, including a link to the license and an indication of any modifications made. This should be done in a reasonable manner, without implying endorsement by the licensor

2. **NonCommercial** — The Algorithm may **NOT** be used for commercial purposes. This includes, but is not limited to, the sale, licensing, or integration of the Software into commercial products or services.

For collaboration or inquiries, please contact us.

## Acknowledgement

This code is based on the [StableBaseline3](https://stable-baselines3.readthedocs.io/en/master/index.html).
