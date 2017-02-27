[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.322785.svg)](https://doi.org/10.5281/zenodo.322785)

Fast and Provably Good Seedings for k-Means using k-MC^2 and AFK-MC^2
===

Introduction
---

The package provides a Cython implementation of the algorithms `k-MC^2` and `AFK-MC^2` described in the two papers:

> **Approximate K-Means++ in Sublinear Time.**
> *Olivier Bachem, Mario Lucic, S. Hamed Hassani and Andreas Krause*.
> In Proc. Conference on Artificial Intelligence (AAAI), 2016.

> **Fast and Provably Good Seedings for k-Means.**
> *Olivier Bachem, Mario Lucic, S. Hamed Hassani and Andreas Krause*.
> To appear in Neural Information Processing Systems (NIPS), 2016.

The implementation is compatible with Python 2.7.

Installation
---
First make sure that `numpy` is installed by running
```
pip install numpy
```

The following command will then install `kmc2` from PyPI:
```
pip install kmc2
```

To install `kmc2` locally from this repository, you may use
```
pip install .
```


Quickstart
---
The `kmc2` function may be used to run the algorithm and obtain a seeding. The data should be provided in a Numpy array or a Scipy CSR matrix.
```python
import kmc2
X = <Numpy array containing the data>
seeding = kmc2.kmc2(X, 5)  # Run k-MC2 with k=5
```

The seeding can then be refined using `MiniBatchKMeans` of `scikit-learn`:
```python
from sklearn.cluster import MiniBatchKMeans
model = MiniBatchKMeans(5, init=seeding).fit(X)
new_centers = model.cluster_centers_
```

Detailed Usage / API
---
The `kmc2` module exposes a single function `kmc2(...)` with all the functionality:
```python
def kmc2(X, k, chain_length=200, afkmc2=True, random_state=None, weights=None):
    """Cython implementation of k-MC2 and AFK-MC2 seeding

    Args:
      X: (n,d)-shaped np.ndarray with data points (or scipy CSR matrix)
      k: number of cluster centers
      chain_length: length of the MCMC chain
      afkmc2: Whether to run AFK-MC2 (if True) or vanilla K-MC2 (if False)
      random_state: numpy.random.RandomState instance or integer to be used as seed
      weights: n-sized np.ndarray with weights of data points (default: uniform weights)


    Returns:
      (k, d)-shaped numpy.ndarray with cluster centers
    """
    ...
```

Tests
---
To run the unittests, use `nose` in the package directory
```
nosetests
```

Feedback / Citation
---
Please send any feedback to Olivier Bachem (<olivier.bachem@inf.ethz.ch>).

If you would like to cite this implementation, please reference the two original papers.

License
---
The software is released under the MIT License as detailed in `kmeans.pyx`.

Acknowledgments
---
This research was partially supported by ERC StG 307036, a Google Ph.D. Fellowship and an IBM Ph.D. Fellowship.
