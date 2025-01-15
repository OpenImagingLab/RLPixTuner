# Goal Conditioned Reinforcement Learning for Photo Finishing Tuning

**NeurIPS 2024**

[Project Page](https://openimaginglab.github.io/RLPixTuner/) | [Video](https://www.youtube.com/watch?v=fFIkc3KHS28)

![img](https://openimaginglab.github.io/RLPixTuner/static/images/teaser.png)


## Installation
```
conda create --name <env> --file requirements.txt
```

## Run

prepare the input-target data pair like:

```
├─ExpertC0001-0-Best-Input.tif
└─ExpertC0001-0-Best-Target.tif
...
```

add the data path into `bash/run.sh`.

```
bash bash/run.sh
```

## Citation
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
