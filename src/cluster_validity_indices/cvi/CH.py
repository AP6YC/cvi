from .common import *

class CH(CVI):
    """
    The stateful information of the Calinski-Harabasz (CH) Cluster Validity Index

    Parameters
    ----------
    dim : int
        Dimensionality of the cluster features.
    """

    # # References
    # 1. L. E. Brito da Silva, N. M. Melton, and D. C. Wunsch II, "Incremental Cluster Validity Indices for Hard Partitions: Extensions  and  Comparative Study," ArXiv  e-prints, Feb 2019, arXiv:1902.06711v1 [cs.LG].
    # 2. T. Calinski and J. Harabasz, "A dendrite method for cluster analysis," Communications in Statistics, vol. 3, no. 1, pp. 1–27, 1974.
    # 3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML]. [Online].
    # 4. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
    # """
    # Calinski-Harabasz (CH) Cluster Validity Index.
    # """

    """
    Parameters
    ----------
    dim : int
        Dimensionality of the cluster features.
    """


    def __init__(self, dim:int):

        super().__init__(dim)

        return
        # print("Hello world!")
        # self.data = []
