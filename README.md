# README
This repository is the official code for our IJCAI 2025 paper: ["A Correlation Manifold Self-Attention Network for EEG Decoding"](https://www.ijcai.org/proceedings/2025/0598.pdf).

If you have any problems, please don't hesitate to contact me.

## Abstract

Riemannian neural networks, which generalize the deep learning paradigm to non-Euclidean geometries, have garnered widespread attention across diverse applications in artificial intelligence. Among these, the representative attention models have been studied on various non-Euclidean spaces to geometrically capture the spatiotemporal dependencies inherent in time series data, e.g., electroencephalography (EEG). Recent studies have highlighted the full-rank correlation matrix as an advantageous alternative to the covariance matrix for data representation, owing to its invariance to the scale of variables. Motivated by these advancements, we propose the Correlation Attention Network (CorAtt) tailored for full-rank correlation matrices and implement it under the permutation-invariant and computationally efficient Off-Log and Log-Scaled geometries, respectively. Extensive evaluations on three benchmarking EEG datasets provide substantial evidence for the effectiveness of our introduced CorAtt.

## Requirements

To install the necessary dependencies by conda, run the following command:
```setup
conda env create CorAtt_env.yml
conda activate CorAtt
```

## Dataset
Please download the datasets and put them in the folder 'data'.

1. MAMEM-SSVEP-II:
   https://www.mamem.eu/results/datasets/
2. BCI-ERN:
    https://www.kaggle.com/competitions/inria-bci-challenge/data

Link to download [data](https://drive.google.com/file/d/1RxN2PWOkYJw-NzyM0vaxdLkina2q-_Rj/view?usp=sharing)

## Training and testing

To train and test the experiments on the Mamem and Bcicha datasets, run this command:

```train and test
python CorrAttMamem.py
python CorrAttBciCha.py
python CorrAttBCI.py
```

## Notes

We encourage you to reproduce the results with different hyperparameter settings. When adapting the model to new tasks, we highlight several implementation details that we found to be particularly important.

In [https://github.com/ChenHu-ML/CorAtt/blob/main/corAtt/CorAtt.py](https://github.com/ChenHu-ML/CorAtt/blob/main/corAtt/CorAtt.py)
1. **The backpropagation of `spd_scaling_finder` in `CorrAttention.lsm_diffeo`.**
   We observed that allowing gradients to propagate through `spd_scaling_finder` does not lead to better performance. In our experiments, disabling backpropagation often yields more stable and better results. Therefore, we wrap this step with `with torch.no_grad()`. However, this behavior may be task-specific, and you can experiment with enabling or disabling gradient propagation.

2. **Number of iterations in `spd_scaling_finder`.**
   In our tasks, a single iteration often provides a sufficient approximation, so we set the default number of iterations to 1. That said, in our earlier experiments, different settings (e.g., 1, 2, or 3 iterations) can lead to better performance under different scenarios. A similar phenomenon is also observed in `hol_plus_finder`, where the optimal number of iterations may vary across tasks.


## Reference
```bib
@inproceedings{hu2025ijcai,
  title={A Correlation Manifold Self-Attention Network for EEG Decoding},
  author={Hu, Chen and Wang, Rui and Song, Xiaoning and Zhou, Tao and Wu, Xiao-Jun and Sebe, Nicu and Chen, Ziheng},
  booktitle={Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
  pages={5372--5380},
  year={2025},
}
```

```bib
@inproceedings{wang2024grassmannian,
  title={A Grassmannian Manifold Self-Attention Network for Signal Classification},
  author={Wang, Rui and Hu, Chen and Chen, Ziheng and Wu, Xiao-Jun and Song, Xiaoning},
  booktitle={Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence},
  pages={5099--5107},
  year={2024}
}

```

