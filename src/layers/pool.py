import numpy as np


class PoolLayer(object):

    def __init__(self,kernel_shape=(3,3),strides=1,mode="max"):
        self.params = {
            'kernel_shape': kernel_shape,
            'strides': strides,
            'mode': mode
        }
        self.cache = dict()


    def forward_propagate(self,X, save_cache=False):
        (num_data_points,prev_height,prev_width,prev_channels)=X.shape
        filter_h,filter_w = self.params.get("kernel")

        n_H = int((prev_height-filter_h)/self.params.get("stride"))+1
        n_W = int((prev_width-filter_w)/self.params.get("stride"))+1
        n_C = prev_channels

        Z_output = np.zeros(shape=(num_data_points,n_H,n_W,n_C))

        for i in range(num_data_points):
            for h in range(n_H):
                vert_start = self.params.get("stride")*h
                vert_end=vert_start+filter_h
                for w in range(n_W):
                    horiz_start=self.params.get("stride")*w
                    horiz_end=horiz_start+filter_w

                    for c in range(n_C):
                        if self.params.get("mode")=="avg":
                            Z_output[i,h,w,c]=np.mean(X[i,vert_start:vert_end,horiz_start:horiz_end,c])
                        elif self.params.get("mode")=="max":
                            Z_output[i,h,w,c]=np.max(X[i,vert_start:vert_end,horiz_start:horiz_end,c])


        if save_cache:
            self.cache["A"] = X

        return Z_output