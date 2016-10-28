# MIT License
#
# Copyright (c) 2016 Olivier Bachem
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Test suite"""
import numpy as np
import warnings
import kmc2
import time
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans


def scenarios():
    """A variety of small-scale problems"""
    rs = np.random.RandomState(0)
    a = rs.randn(500, 2)
    a_sparse = csr_matrix(a)
    lengths = [1, 2, 5, 10]
    for rs in [np.random.RandomState(0), 0, None]:
        for l in lengths:
            for afkmc2 in [True, False]:
                yield dict(X=a, k=5, chain_length=l, afkmc2=afkmc2,
                           random_state=rs, weights=None)
    

def test_scenarios():
    """Test that everything works"""
    for s in scenarios():
        seeding = kmc2.kmc2(**s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # disable sklearn warnings
            model = MiniBatchKMeans(s["k"], init=seeding).fit(s["X"])
        new_centers = model.cluster_centers_
        
                           
def test_sparse_dense():
    """Test sparse / dense consistency"""
    for s in scenarios():
        validate_sparse_dense(**s)

        
def validate_sparse_dense(X, **kwargs):
   """Validate that sparse and dense input gives exactly the same result"""
   kwargs["random_state"] = 1  # important, set the same seed
   X_sparse = csr_matrix(X)
   res = kmc2.kmc2(X, **kwargs)
   res_sparse = kmc2.kmc2(X_sparse, **kwargs)
   np.testing.assert_array_equal(res, res_sparse)


def test_weights():
    """Test weight consistency"""
    for s in scenarios():
        validate_weights(**s)

        
def validate_weights(X, **kwargs):
   """Validate that sparse and dense input gives exactly the same result"""
   kwargs["random_state"] = 1  # important, set the same seed
   # Weights = None
   kwargs["weights"] = None
   res1 = kmc2.kmc2(X, **kwargs) 
   # Weight = np.ones
   kwargs["weights"] = np.ones(X.shape[0])
   res2 = kmc2.kmc2(X, **kwargs)
   np.testing.assert_array_equal(res1, res2)
   # Weight = 1000
   kwargs["weights"] = np.ones(X.shape[0])*1000
   res3 = kmc2.kmc2(X, **kwargs)
   np.testing.assert_array_equal(res2, res3)

   X[0, :] *= 1000
   kwargs["k"] = 5
   # first element has
   kwargs["weights"] = np.ones(X.shape[0])
   kwargs["weights"][0] = 1001
   res4 = kmc2.kmc2(X, **kwargs)
   # one guy with
   X_new = np.vstack((X[[0]*1000,:], X))
   kwargs["weights"] = None
   res5 = kmc2.kmc2(X_new, **kwargs)
   np.testing.assert_array_equal(res4, res5)


def qe(X, centers):
    """Compute the quantization error"""
    a1 = np.sum(np.power(X, 2), axis=1)
    a2 = np.dot(X, centers.T)
    a3 = np.sum(np.power(centers, 2), axis=1)
    dist = - 2*a2 + a3[np.newaxis, :]
    argmin = np.argmin(dist, axis=1)
    mindist = np.min(dist, axis=1) + a1
    error = np.sum(mindist)
    return error
