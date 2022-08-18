import numpy as np

class CVI():

    def __init__(self):
        self.label_map = []
        self.dim = 0
        self.n_samples = 0
        self.mu = np.empty([0])
        self.n = np.empty([0])
        self.v = np.empty([0, 0])
        self.CP = np.empty([0])
        self.SEP = np.empty([0])
        self.G = np.empty([0, 0])
        self.BGSS = 0.0
        self.WGSS = 0.0
        self.n_clusters = 0
        # self.BGSS = np.single()
        # self.WGSS = np.single()
        # self.n_clusters = np.intc()
