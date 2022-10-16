import numpy as np


class Relu(object):
    

    def __init__(self) -> None:
        self.has_units=False
        self.cache={}

    def has_weights(self):
        return self.has_units

    def __str__(self) -> str:
        return "relu layer"

    def forward_propagate(self,Z,save_cache=False):
        if save_cache:
            self.cache["Z"]=Z
        return np.where(Z>=0,Z,0)

    def back_propagate(self,dA):
        Z=self.cache.get("Z")
        return dA*np.where(Z >= 0, 1, 0)


class Softmax:
    def __init__(self):
        self.cache = {}
        self.has_units = False

    def has_weights(self):
        return self.has_units

    def __str__(self) -> str:
        return "softmax layer"

    def forward_propagate(self, Z, save_cache=False):
        if save_cache:
            self.cache['Z'] = Z
        Z_ = Z - Z.max()
        e = np.exp(Z_)
        return e / np.sum(e, axis=0, keepdims=True)

    def back_propagate(self, dA):
        Z = self.cache['Z']
        return dA * (Z * (1 - Z))