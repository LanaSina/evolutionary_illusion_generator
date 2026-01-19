'''

CPPN - Generate Art using Neural Networks

Author - AntixK


Heavily Inspired by - https://github.com/hardmaru/cppn-tensorflow
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Art_Gen(object):

    def initialise_CPPN(self, batch_size = 1, net_size = 32, h_size = 32, x_res = 256, 
                            y_res= 256, scaling = 1.0, RGB = False):

        # Setting Parameters
        self.batch_size = batch_size
        self.net_size = net_size
        self.h_size = h_size
        self.x_res = x_res
        self.y_res = y_res
        self.scaling = scaling

        if RGB == True:
            self.c_dim = 3
        else:
            self.c_dim = 1
        
        self.num_points = x_res * y_res 

        # Configuring Network
        # Lana: this seems useless
        # self.img_batch = np.random.standard_normal(size = (batch_size, x_res, y_res, self.c_dim))
        self.hid_vec = np.random.standard_normal(size =  (batch_size, self.h_size))

        self.x_dat = np.random.standard_normal(size = (batch_size, self.x_res * self.y_res, 1))
        self.y_dat = np.random.standard_normal(size = (batch_size, self.x_res * self.y_res, 1))
        self.r_dat = np.random.standard_normal(size = (batch_size, self.x_res * self.y_res, 1))

        # what is that... inpput for each matrix?
        # mat = np.random.standard_normal(size = (input.shape[1], out_dim)).astype(np.float32)
        self.x_input = np.random.standard_normal(size = (1, net_size)).astype(np.float32)
        self.y_input = np.random.standard_normal(size = (1, net_size)).astype(np.float32)
        self.h_input = np.random.standard_normal(size = (h_size, net_size)).astype(np.float32)
        #self.r_input = np.random.standard_normal(size = (x_res, net_size)).astype(np.float32)

        # what this
        self.h_input_2 = np.random.standard_normal(size = (net_size, net_size)).astype(np.float32)
        self.h_input_3 = np.random.standard_normal(size = (net_size, self.c_dim)).astype(np.float32)
        
    def create_grid(self, x_res = 32, y_res = 32, scaling = 1.0):

        num_points = x_res*y_res
        x_range = np.linspace(-1*scaling, scaling, num = x_res)
        y_range = np.linspace(-1*scaling, scaling, num = y_res)
        x_mat = np.matmul(np.ones((y_res, 1)), x_range.reshape((1, x_res)))
        y_mat = np.matmul(y_range.reshape((y_res, 1)), np.ones((1, x_res)))
        r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
        x_mat = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, num_points, 1)
        y_mat = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, num_points, 1)
        r_mat = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, num_points, 1)

        return x_mat, y_mat, r_mat


    def build_CPPN(self, x_res, y_res, x_dat, y_dat, r_dat, hid_vec):

        num_points = x_res * y_res

        # Scale the hidden vector
        hid_vec_scaled = np.reshape(hid_vec, (self.batch_size, 1, self.h_size)) * \
                            np.ones((num_points, 1), dtype = np.float32) * self.scaling

        # Unwrap the grid matrices      
        x_dat_unwrapped = np.reshape(x_dat, (self.batch_size*num_points, 1))
        y_dat_unwrapped = np.reshape(y_dat, (self.batch_size*num_points, 1))
        r_dat_unwrapped = np.reshape(r_dat, (self.batch_size*num_points, 1))
        h_vec_unwrapped = np.reshape(hid_vec_scaled, (self.batch_size*num_points, self.h_size))


        # Build the network
        self.art_net = self.fully_connected(h_vec_unwrapped, self.net_size, mat = self.h_input) + \
                  self.fully_connected(x_dat_unwrapped, self.net_size, with_bias = False, mat = self.x_input) + \
                  self.fully_connected(y_dat_unwrapped, self.net_size, with_bias = False, mat = self.y_input) + \
                  self.fully_connected(r_dat_unwrapped, self.net_size, with_bias = False)

        # Set Activation function
        out = self.tanh_sig(1)   

        model = np.reshape(out, (self.batch_size, x_res, y_res, self.c_dim))

        return model


    def tanh_sig(self,num_layers = 3):
        h = np.tanh(self.art_net)
        for i in range(num_layers):
            h = np.tanh(self.fully_connected(h, self.net_size, True, self.h_input_2))
        out = self.sigmoid(self.fully_connected(h, self.c_dim, True, self.h_input_3))

        return out


    def sin_tanh_sof(self):
        h = np.tanh(self.art_net)
        h = 0.95*np.sin(self.fully_connected(h,self.net_size))
        h = np.tanh(self.fully_connected(h,self.net_size))
        h = self.soft_plus(self.fully_connected(h,self.net_size))
        h = np.tanh(self.fully_connected(h,self.net_size))
        out = self.soft_plus(self.fully_connected(h,self.c_dim))

        return out

    # not used
    def tanh_sig_sin_sof(self):
        h = np.tanh(self.art_net)
        h = 0.8*np.sin(self.fully_connected(h,self.net_size))
        h = np.tanh(self.fully_connected(h,self.net_size))
        h = self.soft_plus(self.fully_connected(h,self.net_size))
        out = self.sigmoid(self.fully_connected(h, self.c_dim))

        return out
    
    def fully_connected(self, input, out_dim, with_bias = True, mat = None):
        if mat is None:
            mat = np.random.standard_normal(size = (input.shape[1], out_dim)).astype(np.float32)

        result = np.matmul(input, mat)

        if with_bias == True:
            bias = np.random.standard_normal(size =(1, out_dim)).astype(np.float32)
            result += bias * np.ones((input.shape[0], 1), dtype = np.float32)

        return result

    
    def sigmoid(self, x):

        return 1.0 / (1.0 + np.exp(-1* x))  

    def soft_plus(self, x):

        return np.log(1.0 + np.exp(x))  

    def generate(self, x_res = 256, y_res = 256, scaling = 20.0, z = None):

        # Generate Random Key to generate image
        if z is None:
            z = np.random.uniform(low=-1.0, high=1.0, size=(self.batch_size, self.h_size)).astype(np.float32)

        x_dat, y_dat, r_dat = self.create_grid(x_res, y_res, scaling)
        art = self.build_CPPN(x_res, y_res, x_dat, y_dat, r_dat, z)

        return art

def init_pop(n,  net_size = 16, batch_size=1, h_size=32, RGB=True, seed=1):
    np.random.seed(seed)
    nets = [None]*n
    for x in range(n):
        art = Art_Gen()
        art.initialise_CPPN(batch_size, net_size, h_size, RGB = RGB)
        nets[x] = art

    return nets

def Generate_Art(batch_size = 1, net_size = 16, h_size = 8, x_res = 512, y_res= 512, scaling = 10.0, RGB = True, seed = None):

    if seed is not None:
        np.random.seed(seed)

    art = Art_Gen()
    art.initialise_CPPN(batch_size, net_size, h_size,RGB = RGB)

    if RGB == True:
        c_dim = 3
    else:
        c_dim = 1

    image_data = art.generate(x_res, y_res, scaling)

    plt.subplot(1, 1, 1)

    if RGB == False:
        plt.imshow(image_data.reshape(y_res, x_res), cmap='Greys', interpolation='nearest')
    else:
        plt.imshow(image_data.reshape(y_res, x_res, c_dim), interpolation='nearest')
    plt.axis('off')

    img_data = np.array(1-image_data)
    
    if c_dim > 1:
      img_data = np.array(img_data.reshape((y_res, x_res, c_dim))*255.0, dtype=np.uint8)
    else:
      img_data = np.array(img_data.reshape((y_res, x_res))*255.0, dtype=np.uint8)
    im = Image.fromarray(img_data)
    im.save('art.png')
    plt.show()

if __name__ == '__main__':
    #
    Generate_Art(batch_size = 1, net_size = 16, h_size = 32, x_res = 512, y_res= 512, scaling = 10.0, RGB = True, seed = None)
    