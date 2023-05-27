from spn.structure.leaves.parametric.Parametric import Categorical
from spn.algorithms.Marginalization import marginalize
from spn.algorithms.Inference import log_likelihood
import numpy as np

spn = 0.4 * (Categorical(p=[0.2, 0.8], scope=0) *
             (0.3 * (Categorical(p=[0.3, 0.7], scope=1) *
                     Categorical(p=[0.4, 0.6], scope=2))
            + 0.7 * (Categorical(p=[0.5, 0.5], scope=1) *
                     Categorical(p=[0.6, 0.4], scope=2)))) + 0.6 * (Categorical(p=[0.2, 0.8], scope=0) *
             Categorical(p=[0.3, 0.7], scope=1) *
             Categorical(p=[0.4, 0.6], scope=2))

spn_marg = marginalize(spn, [1,2])
test_data = np.array([1.0, 0.0, 1.0]).reshape(-1, 3)

llm = log_likelihood(spn_marg, test_data)
print('Marginalized LL')
print(llm, np.exp(llm))

ll = log_likelihood(spn, test_data)
print('Basic SPN LL')
print(ll, np.exp(ll))