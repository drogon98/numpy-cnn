import numpy as np

from utilities.initializers import he_normal


class FCLayer(object):

    def __init__(self,units=1024) -> None:
        self.units=units
        self.params={}


    def forward_propagate(self,X):
        if "W" not in self.params:
            self.params['W'], self.params['b'] = he_normal((X.shape[0], self.units))
        Z = np.dot(self.params.get('W'), X) + self.params['b']
        return Z