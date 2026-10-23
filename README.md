# The Impact of Overfitting on Privacy Risks in Neural Networks

This project studies how overfitting relates to the vulnerability of neural networks to **membership inference attacks (MIA)**, and whether **differential privacy (DP-SGD)** reduces that vulnerability and at what cost in accuracy.

- Dataset: Fashion-MNIST (50,000 train / 10,000 validation from the official training set, 10,000 official test)
- Model: fully connected MLP (784-256-128-10, 235,146 parameters), Adam, lr 0.001, batch size 128, 30 epochs
- Framework: PyTorch 2.2.2, Opacus 1.4.1

## Research questions

1. How does the degree of overfitting affect the vulnerability of neural networks to membership inference attacks?
2. Can differential privacy reduce the effectiveness of membership inference attacks?
3. What is the trade-off between predictive performance and privacy protection?

## Repository contents

| File | Content |
|---|---|
| `01_baseline_model.ipynb` | Baseline training (3 seeds), generalization gap, loss-based membership inference attack, overfitting-vs-MIA experiment |
| `02_differential_privacy.ipynb` | DP-SGD sweep (epsilon 1, 2, 4, 8; 3 seeds), baseline-vs-DP comparison with the same attack protocol |
| `requirements.txt` | Package versions used for the results |

Running the notebooks creates the folders `data/` (Fashion-MNIST, downloaded automatically) and `results/` (`metrics/`, `figures/`, `checkpoints/`) next to the notebooks.

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks **in this order**: `01_baseline_model.ipynb` first, then `02_differential_privacy.ipynb`. Notebook 01 saves the data split, the configuration, the baseline checkpoints and the metrics that notebook 02 reads. The notebooks run on CPU; the DP sweep (4 epsilons x 3 seeds x 30 epochs) is the slowest part. Set `DP_SEEDS = TRAIN_SEEDS[:1]` in notebook 02 to shorten it.

## Experimental protocol

- **Attack:** loss-based membership inference. The score is `logit(true class) - logsumexp(other logits)` in float64, a monotone transform of the cross-entropy loss that does not saturate to exactly 0 for very confident predictions (float32 loss produces ties there).
- **Members:** the 50,000 training samples. **Non-members:** the official 10,000 test samples. Identical for the baseline and all DP models.
- **Metrics:** ROC-AUC with 95% bootstrap CI, TPR at 1% and 0.1% FPR, attack advantage `max(TPR - FPR)`.
- **Generalization gap:** training minus validation accuracy, both computed in evaluation mode at the end of the epoch.
- **Seeds:** 3 training seeds (42, 43, 44); the data split is fixed.
- **DP-SGD:** Opacus, RDP accountant, delta = 1e-5, max grad norm = 1.0, Poisson sampling. Hyperparameters were not tuned for DP.

## Results (mean over 3 seeds)

### Overfitting and membership inference (baseline checkpoints)

| Epoch | Accuracy gap | MIA ROC-AUC |
|---|---|---|
| 5 | 0.0127 | 0.5108 |
| 10 | 0.0277 | 0.5136 |
| 20 | 0.0471 | 0.5205 |
| 30 | 0.0632 | 0.5262 |

The attack AUC increases together with the generalization gap (Pearson 0.984, Spearman 0.944 over 12 checkpoint/seed points). These points are snapshots of the same three runs, so this is a descriptive association, not a significance test or evidence of causation.

### Baseline vs. DP-SGD

| Model | Test accuracy | Accuracy gap | MIA ROC-AUC | TPR @ 1% FPR | Attack advantage |
|---|---|---|---|---|---|
| Baseline (epsilon = inf) | 88.92% | 0.0632 | 0.5262 | 0.0101 | 0.0740 |
| DP-SGD, epsilon = 8 | 83.43% | 0.0085 | 0.5086 | 0.0108 | 0.0200 |
| DP-SGD, epsilon = 4 | 83.00% | 0.0089 | 0.5085 | 0.0106 | 0.0190 |
| DP-SGD, epsilon = 2 | 82.58% | 0.0078 | 0.5085 | 0.0102 | 0.0183 |
| DP-SGD, epsilon = 1 | 81.70% | 0.0067 | 0.5084 | 0.0109 | 0.0188 |

What the results show, and what they do not:

- DP-SGD costs accuracy (5.5 points at epsilon = 8, 7.2 points at epsilon = 1) and almost removes the generalization gap.
- The loss-based attack is weak even against the baseline (AUC 0.526) and is at chance level in the low-FPR regime (TPR at 1% FPR about 0.01 for every model). The AUC reduction from DP (about 0.018) is therefore small in absolute terms.
- The DP AUC is about 0.508 for all four epsilon values, so no dependence on epsilon is visible with this attack. The cause was not investigated.

## Limitations

- One simple loss-based attack; stronger attacks (e.g. shadow-model / LiRA-style) may reveal more leakage. A low AUC is not a privacy proof.
- One architecture, one dataset, three training seeds, one fixed data split.
- Training duration is only a proxy for overfitting; checkpoints of one run are not independent.
- DP-SGD hyperparameters (learning rate, clipping norm) were not tuned, so part of the accuracy loss may come from untuned settings.
- Opacus ran without secure randomness; this is a research setup, not a production privacy implementation.
- The reported epsilon is the measured value after training and can be slightly above the target (e.g. 8.007 for target 8). Recomputing it with a wider grid of RDP orders gave the same values.

## Planned work

- Move shared code (model, attack functions, training loop) into a `src/` package.
- Evaluate stronger membership inference attacks.
- Add a control experiment to explain why the DP AUC does not change with epsilon.

## References

- Shokri et al. (2017). *Membership Inference Attacks Against Machine Learning Models.* arXiv:1610.05820
- Yeom et al. (2018). *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.* arXiv:1709.01604
- Abadi et al. (2016). *Deep Learning with Differential Privacy.* arXiv:1607.00133
- Carlini et al. (2022). *Membership Inference Attacks From First Principles.* arXiv:2112.03570
- Yousefpour et al. (2021). *Opacus: User-Friendly Differential Privacy Library in PyTorch.* arXiv:2109.12298
- Xiao et al. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747
