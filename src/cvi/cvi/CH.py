import numpy as np

from .common import *

class CH(CVI):
    """
    The stateful information of the Calinski-Harabasz (CH) Cluster Validity Index
    """

    # # References
    # 1. L. E. Brito da Silva, N. M. Melton, and D. C. Wunsch II, "Incremental Cluster Validity Indices for Hard Partitions: Extensions  and  Comparative Study," ArXiv  e-prints, Feb 2019, arXiv:1902.06711v1 [cs.LG].
    # 2. T. Calinski and J. Harabasz, "A dendrite method for cluster analysis," Communications in Statistics, vol. 3, no. 1, pp. 1–27, 1974.
    # 3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML]. [Online].
    # 4. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
    # """
    # Calinski-Harabasz (CH) Cluster Validity Index.
    # """

    def __init__(self):
        """
        CH initialization routine.
        """
        # """
        # Test documentation.
        # """
        super().__init__()

        return
        # print("Hello world!")
        # self.data = []

    def param_inc(self, sample:np.array, label:np.array):
        i_label = self.label_map.get_internal_label(label)

        n_samples_new = self.n_samples + 1
        if not self.mu.any():
            mu_new = sample
            self.setup(sample)
        else:
            mu_new = (1 - 1/n_samples_new) * cvi.mu + (1/n_samples_new) * sample

        if i_label > self.n_clusters:
            n_new = 1
            v_new = sample
            CP_new = 0.0
            G_new = np.zeros(self.dim)