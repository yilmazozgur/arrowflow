# ArrowFlow

**Hierarchical Machine Learning in the Space of Permutations**

[arXiv:2604.04087](https://arxiv.org/abs/2604.04087) | [Read the paper (PDF)](manuscript/ArrowFlow_Yilmaz_2026.pdf)

ArrowFlow is a machine learning architecture that operates entirely in the space of permutations. Its computational units are *ranking filters* -- learned orderings that compare inputs via Spearman's footrule distance and update through permutation-matrix accumulation, a non-gradient rule rooted in displacement evidence.

## Key Ideas

- **Data and parameters are permutations.** No floating-point parameters in the core computation.
- **Learning by displacement accumulation.** Filters reorder based on accumulated positional evidence -- no gradients required.
- **Arrow's impossibility theorem as design principle.** Violations of social-choice fairness axioms (IIA, non-dictatorship, Pareto) serve as inductive biases for nonlinearity, sparsity, and stability.
- **Multi-view ensemble.** Independent networks on diverse random projections, combined by majority vote, compensate for information lost in the ordinal encoding.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from arrowflow import ArrowFlowConfig, run_arrowflow_ensemble
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configure ArrowFlow
config = ArrowFlowConfig(
    no_of_filters=[64, 128],
    layer_types=['sort', 'sort', 'sort'],
    no_of_iters=200,
    no_of_embedding_dim=16,
    poly_expansion=True,
    pol_deg=3,
    n_ensemble_views=7,
    projection_strategy='diverse',
    learning_rate=0.1,
)

# Run
error = run_arrowflow_ensemble(X_train, y_train, X_test, y_test, n_classes=3, config=config)
print(f"Test error: {100*error:.1f}%")
```

## Experiments

Reproduce the paper's experiments:

```bash
# UCI tabular benchmarks
python -m experiments.exp_uci_tabular

# MNIST via PCA
python -m experiments.exp_mnist_pca

# Noise robustness
python -m experiments.exp_noise_robustness

# Gene expression (TCGA)
python -m experiments.exp_gene_expression
```

Results are saved to `experiments/results/`.

## Manuscript

The full paper is in `manuscript/`. Preprint: [arXiv:2604.04087](https://arxiv.org/abs/2604.04087)

## Citation

```bibtex
@article{yilmaz2026arrowflow,
  title={ArrowFlow: Hierarchical Machine Learning in the Space of Permutations},
  author={Yilmaz, Ozgur},
  journal={arXiv preprint arXiv:2604.04087},
  year={2026}
}
```

## License

MIT
