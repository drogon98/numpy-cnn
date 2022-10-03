import numpy as np
from utilities.initializers import glorot_uniform

from utilities.utils import pad_inputs

class ConvLayer(object):
    # Has filters 5*5*3 or 3*3*3
    # The input which is a volume eg 32*32*3
    # each filter is slid over the width and height of the input to produce a 2d output
    # If there are 12 filters they produce a volume with a depth of 12 ie all the individual filters stacked together.
    def __init__(self, filters, kernel_shape=(3, 3), padding='valid', stride=1):
        self.params = {
            'filters': filters,
            'padding': padding,
            'kernel_shape': kernel_shape,
            'stride': stride
        }
        self.cache = {}
        # self.rmsprop_cache = {}
        # self.momentum_cache = {}
        # self.grads = {}
        # self.has_units = True
        # self.name = name
        # self.type = 'conv'


    def conv_single_step(self,input,W,b):
        return np.sum(np.multiply(input,W),b)

    
    def forward_propagate(self,X,save_cache=False):

        (num_data_points, prev_height, prev_width, prev_channels) = X.shape
        filter_h,filter_w = self.params.get("kernel_shape")

        pad_h = 0
        pad_w = 0
        n_H = 0
        n_W = 0

        if "W" not in self.params:
            shape = (filter_h, filter_w, prev_channels, self.params.get('filters'))
            self.params["W"],self.params["b"] = glorot_uniform(shape=shape)

        if self.params.get("padding") == "same":      
            pad_h = int((prev_height+(2*self.params.get("padding"))-filter_h)/self.params.get("stride"))+1
            pad_w = int((prev_width+(2*self.params.get("padding"))-filter_w)/self.params.get("stride"))+1
            # Padding applied ensure output size == input size
            n_H = prev_height
            n_W = prev_width
        else:
            n_H = int((prev_height - filter_h)/self.params.get("stride"))+1 
            n_W = int((prev_width - filter_w)/self.params.get("stride"))+1

        self.params['pad_h'], self.params['pad_w'] = pad_h, pad_w

        Z_output = np.zeros(shape=(num_data_points,n_H,n_W,self.params.get("filters")))

        X_pad = pad_inputs(X,(pad_h,pad_w))

        for i in range(num_data_points):
            x = X_pad[i]
            for h in range(n_H):
                vert_start = self.params.get("stride")*h
                vert_end = filter_h+vert_start
                for w in range(n_W):
                    horiz_start = self.params.get("stride")*w
                    horiz_end = filter_w+horiz_start

                    for c in range(self.params.get("filters")):

                        x_slice = x[vert_start: vert_end, horiz_start: horiz_end, :]

                        Z_output[i,h,w,c] = self.conv_single_step(x_slice,self.params.get("W")[:,:,:,c],self.params.get("b")[:,:,:,c])

        
        if save_cache:
            self.cache["A"] = X

        return Z_output

    
    def back_propagate(self,dZ):
        A = self.cache["A"]
        filter_h,filter_w = self.params.get("kernel")
        pad_h,pad_w = self.params.get("pad_h"),self.params.get("pad_w")
        (num_data_points, prev_height, prev_width, prev_channels) = A.shape
        dA=np.zeros(shape=(num_data_points, prev_height, prev_width, prev_channels))
        self.grads = self.init_cache()
        A_pad = pad_inputs(A, (pad_h, pad_w))
        dA_pad = pad_inputs(dA, (pad_h, pad_w))

        for i in range(num_data_points):
            a_pad = A_pad[i]
            da_pad = dA_pad[i]
            for h in range(prev_height):
                vert_start = self.params.get("stride")*h
                vert_end = vert_start + filter_h

                for w in range(prev_width):
                    horiz_start = self.params.get("stride")*w
                    horiz_end = horiz_start + filter_w

                    for c in range(self.params.get("filters")):
                        a_slice = a_pad[vert_start: vert_end, horiz_start: horiz_end, :]
                        da_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.params['W'][:, :, :, c] * dZ[i, h, w, c]
                        self.grads['dW'][:, :, :, c] += a_slice * dZ[i, h, w, c]
                        self.grads['db'][:, :, :, c] += dZ[i, h, w, c]

            dA[i, :, :, :] = da_pad[pad_h: -pad_h, pad_w: -pad_w, :]
        
        return dA


    def init_cache(self):
        cache = dict()
        cache["dW"] = np.zeros_like(self.params.get("W"))
        cache["db"] = np.zeros_like(self.params.get("b"))
        return cache


