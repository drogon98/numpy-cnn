import numpy as np
import pickle

from os import path,makedirs,remove
from layers.base import Layer

from utilities.initializers import glorot_uniform
from utilities.settings import get_layer_num, inc_layer_num
from utilities.utils import pad_inputs


class ConvLayer(Layer):
    def __init__(self, filters, kernel_shape=(3, 3), padding='valid', strides=1,name=None):
        self.params = {
            'filters': filters,
            'padding': padding,
            'kernel_shape': kernel_shape,
            'strides': strides
        }
        self.cache = {}
        self.rmsprop_cache = {}
        self.momentum_cache = {}
        self.grads = {} 
        self.has_units = True
        self.name = name
        self.type = 'conv'

    def has_weights(self):
        return self.has_units


    def __str__(self) -> str:
        return "conv layer"


    def conv_single_step(self,input,W,b):
        return np.sum(np.multiply(input, W)) + float(b)

    def get_pad_value(self,width=False):
        # f, kernel width and height is often odd to have a center 
        f = self.params.get("kernel_shape")[0]                 
        return int((f-1)/2)

    def get_strided_pad_value(self,input_dim):
        # f, kernel width and height is often odd to have a center
        strides= self.params.get("strides")
        f = self.params.get("kernel_shape")[0]                    
        return int((strides(input_dim-1)-input_dim+f)/2)

    
    def forward_propagate(self,X,save_cache=False):

        if self.name is None:
            self.name = '{}_{}'.format(self.type, get_layer_num(self.type))
            inc_layer_num(self.type)

        (num_data_points, input_height, input_width, input_channels) = X.shape
        filter_height,filter_width = self.params.get("kernel_shape")

        if "W" not in self.params:
            shape = (filter_height, filter_width, input_channels, self.params.get('filters'))
            self.params["W"],self.params["b"] = glorot_uniform(shape=shape)

        padded_height = 0
        padded_width = 0
        output_height = 0
        output_width = 0

        strides = self.params.get("strides")

        if strides!=1:
            # Strided convolution
            if self.params.get("padding") == "same":      
                padded_height = int((input_height+(2*self.get_strided_pad_value(input_height)) - filter_height)/strides)+1
                padded_width = int((input_width+(2*self.get_strided_pad_value(input_width)) - filter_width)/strides)+1
                output_height = input_height
                output_width = input_width
            else:
                output_height = int((input_height - filter_height)/strides)+1 
                output_width = int((input_width - filter_width)/strides)+1
        else:
            if self.params.get("padding") == "same":      
                padded_height = int(input_height+(2*self.get_pad_value()) - filter_height +1)
                padded_width = int(input_width+(2*self.get_pad_value()) - filter_width +1)
                output_height = input_height
                output_width = input_width
            else:
                output_height = int(input_height - filter_height+1) 
                output_width = int(input_width - filter_width+1)
        

        # self.params['pad_h'], self.params['pad_w'] = pad_h, pad_w

        Z_output = np.zeros(shape=(num_data_points,output_height,output_width,self.params.get("filters")))

        X_pad = pad_inputs(X,(padded_height,padded_width))

        for i in range(num_data_points):
            x = X_pad[i]
            for h in range(output_height):
                vert_start = self.params.get("strides")*h
                vert_end = filter_height+vert_start
                for w in range(output_width):
                    horiz_start = self.params.get("strides")*w
                    horiz_end = filter_width+horiz_start

                    for c in range(self.params.get("filters")):

                        x_slice = x[vert_start: vert_end, horiz_start: horiz_end, :]

                        Z_output[i,h,w,c] = self.conv_single_step(x_slice,self.params.get("W")[:,:,:,c],self.params.get("b")[:,:,:,c])

        
        if save_cache:
            # Save for back propagation
            self.cache["A"] = X

        return Z_output

    
    def back_propagate(self,dZ):
        # A layer has its activations
        A = self.cache["A"]
        filter_h,filter_w = self.params.get("kernel_shape")
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
    

    def momentum(self, beta=0.9):
        if not self.momentum_cache:
            self.momentum_cache = self.init_cache()
        self.momentum_cache['dW'] = beta * self.momentum_cache['dW'] + (1 - beta) * self.grads['dW']
        self.momentum_cache['db'] = beta * self.momentum_cache['db'] + (1 - beta) * self.grads['db']

    def rmsprop(self, beta=0.999, amsprop=True):
        if not self.rmsprop_cache:
            self.rmsprop_cache = self.init_cache()

        new_dW = beta * self.rmsprop_cache['dW'] + (1 - beta) * (self.grads['dW']**2)
        new_db = beta * self.rmsprop_cache['db'] + (1 - beta) * (self.grads['db']**2)

        if amsprop:
            self.rmsprop_cache['dW'] = np.maximum(self.rmsprop_cache['dW'], new_dW)
            self.rmsprop_cache['db'] = np.maximum(self.rmsprop_cache['db'], new_db)
        else:
            self.rmsprop_cache['dW'] = new_dW
            self.rmsprop_cache['db'] = new_db


    def apply_grads(self,learning_rate=0.001, l2_penalty=1e-4, optimization='adam', epsilon=1e-8,
                    correct_bias=False, beta1=0.9, beta2=0.999, iter=999):
        if optimization == "adam":
            if correct_bias:
                W_first_moment = self.momentum_cache['dW'] / (1 - beta1 ** iter)
                b_first_moment = self.momentum_cache['db'] / (1 - beta1 ** iter)
                W_second_moment = self.rmsprop_cache['dW'] / (1 - beta2 ** iter)
                b_second_moment = self.rmsprop_cache['db'] / (1 - beta2 ** iter)
            else:
                W_first_moment = self.momentum_cache['dW']
                b_first_moment = self.momentum_cache['db']
                W_second_moment = self.rmsprop_cache['dW']
                b_second_moment = self.rmsprop_cache['db']

            W_learning_rate = learning_rate / (np.sqrt(W_second_moment) + epsilon)
            b_learning_rate = learning_rate / (np.sqrt(b_second_moment) + epsilon)

            self.params['W'] -= W_learning_rate * (W_first_moment + l2_penalty * self.params['W'])
            self.params['b'] -= b_learning_rate * (b_first_moment + l2_penalty * self.params['b'])

        else:
            self.params["W"] -= learning_rate*(self.grads.get("dW")+l2_penalty*self.params.get("W"))
            self.params["b"] -= learning_rate*(self.grads.get("db")+l2_penalty*self.params.get("b"))


