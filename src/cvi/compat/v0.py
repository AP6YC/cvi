"""
Incremental CVI variants.
"""

# Custom imports
import numpy as np

from typing import Optional


class Cluster:
    """
    Base cluster class.
    """

    def __init__(self, radial: bool = True, box: bool = False):
        """
        Cluster constructor.

        Parameters
        ----------
        radial : bool
            TODO
        box : bool
            TODO
        """

        self.radial = radial
        self.box = box
        self.len = 0
        self.center = 0
        self.bounding_box = []
        self.points = []
        self.indicies = []
        self.dim = 0

    def add_point(self, x: np.ndarray, i: Optional[int] = None):
        """
        Adds a point to the cluster.

        Parameters
        ----------
        x : np.ndarray
            TODO
        i : Optional[int]
            TODO
        """

        if not self.points:
            self.dim = len(x)
        assert self.dim == len(x)
        if self.radial:
            new_center = (self.center * self.len + x) / (self.len + 1)
            self.center = new_center
        if self.box:
            assert np.max(x) <= 1
            assert np.min(x) >= 0
            cc_x = np.append(x, 1-x)
            if self.bounding_box:
                self.bounding_box = np.minimum(self.bounding_box, cc_x)
            else:
                self.bounding_box = cc_x

        self.points.append(x)
        if i is not None:
            self.indicies.append(i)
        self.len += 1

    def min_distance(self, x: np.ndarray, norm_ord: Optional[int] = 2):
        """
        Calculates the minimum distance to the nearest cluster.

        Parameters
        ----------
        x : np.ndarray
            TODO
        norm_ord : Optional[int]
            TODO
        """

        min_d = np.inf
        for y in self.points:
            d_y = np.linalg.norm((x - y), norm_ord)
            min_d = np.minimum(min_d, d_y)
        return min_d

    def avg_distance(self, x: np.ndarray, norm_ord: Optional[int] = 2) -> float:
        """
        Gets the average distance of a point.

        Parameters
        ----------
        x : np.ndarray
            TODO
        norm_ord : Optional[int]
            TODO
        """

        avg_d = 0
        for y in self.points:
            avg_d += np.linalg.norm((x - y), norm_ord)
        avg_d /= self.len
        return avg_d

    def center_distance(self, x: np.ndarray, norm_ord: Optional[int] = 2):
        """
        Gets the center distance of a point.

        Parameters
        ----------
        x : np.ndarray
            TODO
        norm_ord : Optional[int]
            TODO
        """

        return np.linalg.norm(self.center - x, norm_ord)

    def within_bounding_box(self, x: np.ndarray):
        """
        Checks if a point is within a bounding box.

        Parameters
        ----------
        x : np.ndarray
            TODO
        """

        return_val = np.array_equal(
            np.minimum(x, self.bounding_box),
            self.bounding_box
        )
        return return_val


class Clusters(list):
    """
    Base clusters class.
    """
    def from_list(self, X, C):
        for x, c in zip(X, C):
            if c >= self.__len__():
                self.append(Cluster)
            self[c].add_point(c, c)


def norm22(x: np.ndarray):
    """
    Calculates the 2-norm squared.

    Parameters
    ----------
    x : np.ndarray
        The vector to norm and square.
    """
    return np.linalg.norm(x, 2) ** 2


def CP_update(
    x: np.ndarray,
    v_old: np.ndarray,
    n_old: np.ndarray,
    g_old: Optional[np.ndarray] = None,
    cp_old: float = 0,
    n_new: Optional[np.ndarray] = None,
    v_new: Optional[np.ndarray] = None,
    p: int = 2,
    q: int = 2
):
    """
    CP and optionally G, v, and n update.

    Parameters
    ----------
    x : np.ndarray
        TODO
    v_old : np.ndarray
        TODO
    n_old : np.ndarray,
        TODO
    g_old : Optional[np.ndarray] = None
        TODO
    cp_old : float = 0
        TODO
    n_new : Optional[np.ndarray] = None
        TODO
    v_new : Optional[np.ndarray] = None
        TODO
    p : int = 2
        TODO
    q : int = 2
        TODO
    """

    if g_old is None:
        g_old = np.zeros_like(x)
    if n_new is None:
        n_new = n_old + 1
    if v_new is None:
        v_new = v_old + (x - v_old) / n_new
    delta_v = v_old - v_new
    z = x - v_new
    g_new = g_old + z + n_old * delta_v
    cp_new = (
        cp_old
        + np.linalg.norm(z, q) ** p
        + n_old * np.linalg.norm(delta_v, q) ** p
        + np.sqrt(2 * np.dot(delta_v, g_old)) ** p
    )

    return cp_new, g_new, v_new, n_new


def cluster_center_update(
    x: np.ndarray,
    v_old: np.ndarray,
    n_old: np.ndarray
):
    """
    Updates the cluster center incrementally.

    Parameters
    ----------
    x : np.ndarray
        TODO
    v_old : np.ndarray
        TODO
    n_old : np.ndarray
        TODO
    """

    n_new = n_old + 1
    v_new = v_old + (x - v_old) / n_new
    return v_new, n_new


class iXB:
    """
    Incremental Xie-Beni (XB) CVI.
    """

    def __init__(self):
        """
        XB constructor.
        """

        self.min_v = np.inf
        self.min_v_i = []
        self.N = 0

        self.WGSS = 0
        self.WGSS_i = []
        self.output = 0

        self.cluster_centers = []
        self.cluster_sizes = []
        self.g = None

    def update(self, x: np.ndarray, c_i: int):
        """
        XB incremental update method.

        Parameters
        ----------
        x : np.ndarray
            TODO
        c_i : int
            TODO
        """

        self.N += 1
        if c_i == len(self.cluster_sizes):
            self.cluster_centers.append(x)
            self.cluster_sizes.append(1)
            self.WGSS_i.append(0)
            self.min_v_i.append(np.inf)
        elif c_i > len(self.cluster_sizes):
            raise ValueError('Invalid Cluster Ordering')
        else:
            self.WGSS -= self.WGSS_i[c_i]
            (
                self.WGSS_i[c_i],
                self.g,
                self.cluster_centers[c_i],
                self.cluster_sizes[c_i]
            ) = CP_update(
                    x,
                    self.cluster_centers[c_i],
                    self.cluster_sizes[c_i],
                    self.g,
                    self.WGSS_i[c_i]
            )

        # self.WGSS_i[c_i] = CP(self.clusters[c_i].center,self.clusters[c_i],2,2)
        self.WGSS += self.WGSS_i[c_i]
        self.min_v_i[c_i] = np.inf
        for j in range(len(self.cluster_centers)):
            if j != c_i:
                self.min_v_i[c_i] = np.minimum(
                    self.min_v_i[c_i],
                    norm22(self.cluster_centers[c_i]-self.cluster_centers[j])
                )

        self.min_v = min([self.min_v, self.min_v_i[c_i]])

        self.output = (self.WGSS / self.N) / self.min_v
        return self.output


class iPS:
    """
    Incremental Partition Separation (PS).
    """

    def __init__(self):
        """
        PS constructor.
        """

        self.mean_cluster_center = None
        self.max_cluster_size = 0

        self.output = 0
        self.PS_i = []

        self.cluster_centers = []
        self.cluster_sizes = []

    def update(self, x: np.ndarray, c_i: int):
        """
        PS incremental update.

        Parameters
        ----------
        x : np.ndarray
            TODO
        c_i : int
            TODO
        """

        if c_i >= len(self.cluster_centers):
            self.cluster_centers.append(x)
            self.cluster_sizes.append(1)
            self.PS_i.append(0)
        else:
            (
                self.cluster_centers[c_i],
                self.cluster_sizes[c_i]
            ) = cluster_center_update(
                x,
                self.cluster_centers[c_i],
                self.cluster_sizes[c_i]
            )
        self.max_cluster_size = (
            np.maximum(self.max_cluster_size, self.cluster_sizes[c_i])
        )
        b_ = ([
            norm22(self.cluster_centers[c_i] - self.cluster_centers[c_j])
            for c_j in range(len(self.cluster_centers))
            if c_j != c_i
        ])
        if b_:
            b = np.min(b_)
            self.mean_cluster_center = (
                np.sum(self.cluster_centers) / len(self.cluster_centers)
            )
            Bt = (
                np.sum(
                    norm22(cc - self.mean_cluster_center)
                    for cc in self.cluster_centers
                ) / len(self.cluster_centers)
            )
            self.PS_i[c_i] = (
                self.cluster_sizes[c_i] / self.max_cluster_size
                - np.exp(-b / Bt)
            )
        else:
            self.PS_i[c_i] = 0

            self.output = np.sum(self.PS_i)
        return self.output


class iCH:
    """
    Incremental Calinski-Harabasz (CH) CVI.
    """

    def __init__(self):
        """
        CH constructor.
        """

        self.clusters = Clusters()

        self.WGSS = 0
        self.WGSS_i = []
        self.BGSS = 0
        self.BGSS_i = []
        self.output = 0
        self.data_center = 0
        self.N = 0

        self.cluster_centers = []
        self.cluster_sizes = []
        self.g = None

    def update(self, x: np.ndarray, c_i: int):
        """
        CH incremental update.

        Parameters
        ----------
        x : np.ndarray
            TODO
        c_i : int
            TODO
        """

        self.N += 1
        self.data_center = (
            ((self.N - 1) * self.data_center + x) / self.N
        )
        if c_i == len(self.clusters):
            self.cluster_centers.append(x)
            self.cluster_sizes.append(1)
            self.WGSS_i.append(0)
            self.BGSS_i.append(0)
        elif c_i > len(self.cluster_sizes):
            raise ValueError('Invalid Cluster Ordering')
        else:
            self.WGSS -= self.WGSS_i[c_i]
            (
                self.WGSS_i[c_i],
                self.g,
                self.cluster_centers[c_i],
                self.cluster_sizes[c_i]
            ) = CP_update(
                x,
                self.cluster_centers[c_i],
                self.cluster_sizes[c_i],
                self.g,
                self.WGSS_i[c_i]
            )
            self.WGSS += self.WGSS_i[c_i]

        self.BGSS -= self.BGSS_i[c_i]
        self.BGSS_i[c_i] = (
            self.cluster_sizes[c_i]
            * norm22(self.cluster_centers[c_i] - self.data_center)
        )
        self.BGSS += self.BGSS_i[c_i]

        if (len(self.cluster_centers) > 1) and (self.N - len(self.cluster_centers) > 0):
            self.output = (
                (self.BGSS / (len(self.cluster_centers) - 1))
                / (self.WGSS / (self.N - len(self.cluster_centers)))
            )
        else:
            self.output = 0
        return self.output


class iGD:
    """
    Incremental Generalized Dunn's (GD) Index.
    """

    def __init__(self, t: int = 43):
        """
        GD constructor.

        Parameters
        ----------
        t : int
            GD type, either 43 or 53.
        """

        assert (t == 43 or t == 53)

        self.t = t
        self.d = np.ones((0, 0))*np.inf
        self.D = []
        self.output = 0

        self.cluster_centers = []
        self.cluster_sizes = []
        self.g = None
        self.CP = []

    def update(self, x: np.ndarray, c_i: int):
        """
        GD incremental update.

        Parameters
        ----------
        x : np.ndarray
            TODO
        c_i : int
            TODO
        """

        if c_i == len(self.cluster_centers):
            self.cluster_centers.append(x)
            self.cluster_sizes.append(1)
            self.CP.append(0)
            self.d = np.pad(
                self.d,
                [(0, 1), (0, 1)],
                mode='constant',
                constant_values=np.inf
            )
            self.D.append(-np.inf)
        elif c_i > len(self.cluster_sizes):
            raise ValueError('Invalid Cluster Ordering')
        else:
            (
                self.CP[c_i],
                self.g,
                self.cluster_centers[c_i],
                self.cluster_sizes[c_i]
            ) = CP_update(
                x,
                self.cluster_centers[c_i],
                self.cluster_sizes[c_i],
                self.g,
                self.CP[c_i],
                p=1
            )
            if self.t == 43:
                for j in range(len(self.cluster_centers)):
                    if c_i != j:
                        self.d[c_i, j] = norm22(
                            self.cluster_centers[c_i]
                            - self.cluster_centers[j]
                        )
                        self.d[j, c_i] = self.d[c_i, j]
            else:
                for j in range(len(self.cluster_centers)):
                    if c_i != j:
                        self.d[c_i, j] = (
                            (self.CP[c_i] + self.CP[j])
                            / (self.cluster_sizes[c_i] + self.cluster_sizes[j])
                        )
                        self.d[j, c_i] = self.d[c_i, j]
            self.D[c_i] = (
                2 * self.CP[c_i]
                / self.cluster_sizes[c_i].len
            )
            self.output = np.min(self.d) / np.max(self.D)

        return self.output


class iGD43(iGD):
    """
    Generalized Dunn's Index 43 (GD43) CVI.
    """

    def __init__(self):
        super().__init__(43)


class iGD53(iGD):
    """
    Generalized Dunn's Index 53 (GD53)
    """

    def __init__(self):
        super().__init__(53)


class iSIL:
    """
    Incremental Silhouette (SIL) index.
    """

    def __init__(self):
        """
        SIL constructor.
        """

        self.sc = []
        self.b = np.ones((0, 0)) * np.inf
        self.output = 0
        self.N = 0

        self.cluster_centers = []
        self.cluster_sizes = []
        self.CP = []
        self.g = []

        self.sij = np.zeros((0, 0))

    def update(self, x: np.ndarray, i: int):
        """
        SIL incremental update.

        Parameters
        ----------
        x : np.ndarray
            TODO
        i : int
            TODO
        """
        self.CP[i] += np.linalg.norm(x) ** 2
        self.g[i] += x

    def s_ij_new(
        self,
        x: np.ndarray,
        i: int,
        j: int,
        J: int,
        nj: int,
        nj_old: int,
        cpj_old: float,
        vj: np.ndarray,
        vi: np.ndarray,
        gj_old: np.ndarray,
        sij_old: float
    ):
        """
        Computes the updated S_ij entry.

        Parameters
        ----------
        x : np.ndarray
            TODO
        i : int
            TODO
        j : int
            TODO
        J : int
            TODO
        nj : int
            TODO
        nj_old : int
            TODO
        cpj_old : float
            TODO
        vj : np.ndarray
            TODO
        vi : np.ndarray
            TODO
        gj_old : np.ndarray
            TODO
        sij_old : float
            TODO
        """

        zi = x - vi
        zj = x - vj
        if i != J:
            if j == J:
                local_return = (
                    (1 / nj) * (
                        cpj_old
                        + np.linalg.norm(zi) ** 2
                        + nj * np.linalg.norm(vi) ** 2
                        - 2 * np.dot(vi, gj_old)
                    )
                )
                return local_return
            else:
                return sij_old
        else:
            if j == J:
                local_return = (
                    (1 / nj) * (
                        cpj_old
                        + np.linalg.norm(zj) ** 2
                        + nj_old * np.linalg.norm(vj) ** 2
                        - 2 * np.dot(vj, gj_old)
                    )
                )
                return local_return
            else:
                local_return = (
                    (1 / nj) * (
                        cpj_old
                        + nj * np.linalg.norm(vi) ** 2
                        - 2 * np.dot(vi, gj_old)
                    )
                )
                return local_return

    def s_ij_new_cluster(
        self,
        x: np.ndarray,
        i: int,
        j: int,
        J: int,
        nj_old: int,
        cpj_old: float,
        vi_new: np.ndarray,
        vi_old: np.ndarray,
        gj_old: np.ndarray,
        sij_old: float
    ):
        """
        TODO

        Parameters
        ----------
        x : np.ndarray
            TODO
        i : int
            TODO
        j : int
            TODO
        J : int
            TODO
        nj_old : int
            TODO
        cpj_old : float
            TODO
        vi_new : np.ndarray
            TODO
        vi_old : np.ndarray
            TODO
        gj_old : np.ndarray
            TODO
        sij_old : float
            TODO
        """

        if i != J:
            if j == J:
                local_return = (
                    np.linalg.norm(x) ** 2
                    + np.linalg.norm(vi_old) ** 2
                    - 2 * np.dot(vi_old, x)
                )
                return local_return
            else:
                return sij_old
        else:
            if j == J:
                return 0
            else:
                local_return = (
                    (1 / nj_old) * (
                        cpj_old
                        + nj_old * np.linalg.norm(vi_new) ** 2
                        - 2 * np.dot(vi_new, gj_old)
                    )
                )
                return local_return

    def sci(self,i, J):
        A = min(self.sij[i,l] for l in range(len(self.cluster_centers)) if l != J) - self.sij[i,J]
        B = max(self.sij[i,J],max(self.sij[i,l] for l in range(len(self.cluster_centers)) if l != J))
        return A/B

    def update(self,x,c_i):
        if c_i == len(self.cluster_sizes):
            self.cluster_centers.append(x)
            self.cluster_sizes.append(1)
            self.sc.append(0)
            self.b = np.pad(self.b, [(0, 1), (0, 1)], mode='constant', constant_values=np.inf)
            self.sij = np.pad(self.b, [(0, 1), (0, 1)], mode='constant', constant_values=0)
            self.CP.append(np.linalg.norm(x) ** 2)
            self.g.append(x)
        elif c_i > len(self.cluster_sizes):
            raise ValueError('Invalid Cluster Ordering')
        self.N += 1
        self.cluster_centers[c_i], self.cluster_sizes[c_i] = cluster_center_update(x, self.cluster_centers[c_i], self.cluster_sizes[c_i])
        for i in range(len(self.cluster_sizes)):
            for j in range(len(self.cluster_sizes)):
                self.sij[i, j] = self.s_ij_new(x, i, j, c_i, self.cluster_sizes[j], self.cluster_sizes[j]-1, self.CP[j], self.cluster_centers[j], self.cluster_centers[i], self.g[j], self.sij[i, j])
        self.cp_update(x, c_i)
        self.output = (1/len(self.cluster_centers))*sum([self.sci(i,c_i) for i in range(len(self.cluster_centers))])
        return self.output

class iDB:
    def __init__(self):

        self.R = []
        self.S = []
        self.M = np.zeros((0,0))
        self.p = 1
        self.q = 1
        self.output = 0
        self.N = 0

        self.cluster_centers = []
        self.cluster_sizes = []
        self.g = None
        self.CP = []

    def update(self,x,c_i):
        self.N += 1
        if c_i == len(self.cluster_centers):
            self.cluster_centers.append(x)
            self.cluster_sizes.append(1)
            self.R.append(0)
            self.S.append(0)
            self.M = np.pad(self.M, [(0, 1), (0, 1)], mode='constant', constant_values=0)
        elif c_i > len(self.cluster_sizes):
            raise ValueError('Invalid Cluster Ordering')
        else:
            self.CP[c_i], self.g, self.cluster_centers[c_i], self.cluster_sizes[c_i] = CP_update(x,self.cluster_centers[c_i],self.cluster_sizes[c_i], self.g,self.CP[c_i], p=self.p, q=self.q)
            self.S[c_i] = ((1/self.cluster_sizes[c_i])*self.CP[c_i])**(1/self.q)
            self.R[c_i] = 0
            for j in range(len(self.cluster_sizes)):
                if j != c_i:
                    self.M[c_i, j] = 0
                    for t in range(len(x)):
                        if self.cluster_sizes[j] > 0:
                            self.M[c_i, j] = abs(self.cluster_centers[c_i][t]-self.cluster_centers[j][t])**self.p
                    self.M[c_i, j] = self.M[c_i, j] ** (1./self.p)
                    self.M[j, c_i] = self.M[c_i, j]
                    if self.cluster_sizes[j] > 0:
                        self.R[c_i] = max(self.R[c_i], (self.S[c_i] + self.S[j]) / self.M[c_i,j])

            self.output = np.sum(self.R)/sum([clen > 0 for clen in self.cluster_sizes])
        return self.output


def iCVI(name):
    if name == 'iDB':
        return iDB()
    elif name == 'iSIL':
        return iSIL()
    elif name == 'GD43':
        return iGD43()
    elif name == 'iGD53':
        return iGD53()
    elif name == 'iCH':
        return iCH()
    elif name == 'iPS':
        return iPS()
    elif name == 'iXB':
        return iXB()
    else:
        raise ValueError('iCVI {} not implemented'.format(name))
