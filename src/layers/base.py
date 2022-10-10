import pickle
import numpy as np

from os import path,makedirs,remove

from utilities.settings import get_layer_num, inc_layer_num



class Layer(object):

    def init_cache(self):
        cache = dict()
        cache['dW'] = np.zeros_like(self.params.get('W'))
        cache['db'] = np.zeros_like(self.params.get('b'))
        return cache

    def save_weights(self,dump_path):
        dump_cache = {
            'cache': self.cache,
            'grads': self.grads,
            'momentum': self.momentum_cache,
            'rmsprop': self.rmsprop_cache
        }
        save_path = path.join(dump_path,f"{self.name}.pickle")
        makedirs(path.dirname(save_path),exist_ok=True)
        # remove(save_path)
        with open(save_path,"wb") as d:
            pickle.dump(dump_cache,d)

    
    def load_weights(self,dump_path):
        if self.name is None:
            self.name = '{}_{}'.format(self.type, get_layer_num(self.type))
            inc_layer_num(self.type)

            
        read_path = path.join(dump_path, self.name+'.pickle')
        with open(read_path, 'rb') as r:
            dump_cache = pickle.load(r)
        self.cache = dump_cache.get('cache')
        self.grads = dump_cache.get('grads')
        self.momentum_cache = dump_cache.get('momentum')
        self.rmsprop_cache = dump_cache.get('rmsprop')