# MADNet: Moment-constrained Automatic Debiasing Networks

Code to accompany the paper: _Automatic debiasing of neural networks via moment-constrained learning_.

## Set up

```shell
python -m venv .venv
source .venv/bin/activate
python -m pip install -r paper/requirements.txt
python -m pip install -e '.[paper]'
```

To replicate the numerical experiments

```shell
python paper/benchmark.py --dataset ihdp --numruns 1000 --config paper/madnet_ate.yaml
python paper/benchmark.py --dataset bhp --numruns 200 --config paper/madnet_ade.yaml
```

## Semi-synthetic datasets

### BHP - Gasoline demand data from Blundell et al. (2017).

This data is shared in this repo under the public domain licence described at the [Harvard dataverse.](https://dataverse.harvard.edu/dataset.xhtml;jsessionid=ab284f8afb3805aad6f8c6b9ddca?persistentId=doi%3A10.7910%2FDVN%2F0YALNP&version=&q=&fileTypeGroupFacet=%22Data%22&fileAccess=&fileTag=&fileSortField=&fileSortOrder=)

### IHDP - Infant Health and Development Program

Obtained directly from the [RieszLearning](https://github.com/victor5as/RieszLearning/) repository. We use this data under the MIT Licence of that repository.

## Description

Repo overview:

```
├── paper (figures and benchmarking scripts)
├── madnet
│   ├── datasets (for benchmarking numerical experiments)
│   ├── estimators (learner implementations in jax + equinox)
│   ├── model_selection (cross-fitting etc.)
└── └── estimands (some common average moment functionals)
```

Provides implementations of:

- MADNet (Proposed)
- RieszNet (Chernozhukov et al. 2022)
- DragonNet (Shi et al. 2019)

## Reproducibility

The package versions that were used to obtain the RieszNet IHDP values in Table 1 can be found in `requirements-rieszlearning.txt`.

1. clone the [RieszLearning](https://github.com/victor5as/RieszLearning) repo.
2. make a virtual environment and run `python -m pip install -r requirements-rieszlearning.txt`.
3. run `RieszNet_IHDP.ipynb` notebook and use the output in `results/IHDP/RieszNet/MAE/IHDP_MAE_NN.tex`.

## References

- Chernozhukov, V., Newey, W., Quintas-Martı́nez, V. M., & Syrgkanis, V. (2022). RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets and Random Forests. _Proceedings of the 39th International Conference on Machine Learning_, 3901–3914. https://proceedings.mlr.press/v162/chernozhukov22a.html
- Shi, C., Blei, D., & Veitch, V. (2019). Adapting Neural Networks for the Estimation of Treatment Effects. _Advances in Neural Information Processing Systems_, 32. https://proceedings.neurips.cc/paper/2019/hash/8fb5f8be2aa9d6c64a04e3ab9f63feee-Abstract.html
