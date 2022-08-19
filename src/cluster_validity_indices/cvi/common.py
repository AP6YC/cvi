import numpy as np

class CVI():
    """
    Superclass containing elements shared between all CVIs.

    Parameters
    ----------
    dim : int
        The dimensionality of the cluster features.
    """

    def __init__(self, dim:int):
        """
        Test documentation.
        """
        self.label_map = []
        self.dim = 0
        self.n_samples = 0
        self.mu = np.empty([dim])
        self.n = np.empty([dim])
        self.v = np.empty([dim, 0])
        self.CP = np.empty([dim])
        self.SEP = np.empty([dim])
        self.G = np.empty([dim, 0])
        self.BGSS = 0.0
        self.WGSS = 0.0
        self.n_clusters = 0
        # self.BGSS = np.single()
        # self.WGSS = np.single()
        # self.n_clusters = np.intc()
        return

    # def __init__(self, dim:int):
    #     self.label_map = []
    #     self.dim = 0
    #     self.n_samples = 0
    #     self.mu = np.empty([0])
    #     self.n = np.empty([0])
    #     self.v = np.empty([0, 0])
    #     self.CP = np.empty([0])
    #     self.SEP = np.empty([0])
    #     self.G = np.empty([0, 0])
    #     self.BGSS = 0.0
    #     self.WGSS = 0.0
    #     self.n_clusters = 0
    #     # self.BGSS = np.single()
    #     # self.WGSS = np.single()
    #     # self.n_clusters = np.intc()

class LabelMap():
    """
    Internal map between labels and the incremental CVI categories.
    """

    def __init__(self):
        self.map = dict()
        return

    def get_internal_label(self, label:int) -> int:
        """
        Gets the internal label and updates the label map if the label is new.

        """
        if label in self.map:
            internal_label = label
        else:
            internal_label = len(self.map.items) + 1
            self.map.items[label] = internal_label

        return internal_label