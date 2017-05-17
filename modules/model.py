import tensorflow as tf

class Model:

    def __init__(self, name, features):
        self.name = name
        self.outputs = [features]
        self.variables = []

    def get_output(self):
        """Returns the output from the topmost layer of the network"""
        return self.outputs[-1]

    def _get_layer_str(self):
        return '%s_L%03d' % (self.name, len(self.outputs)+len(self.variables))

    def weight_variable(self, shape, stddev=0.1):
        with tf.variable_scope(self._get_layer_str()):
            initial = tf.truncated_normal(shape, stddev=stddev)
            weight = tf.get_variable('weight', initializer=initial)
        self.variables.append(weight)
        return weight

    def _glorot_initializer(self, prev_units, num_units, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
        
        stddev  = np.sqrt(stddev_factor / np.sqrt(prev_units*num_units))
        return tf.truncated_normal([prev_units, num_units],
                                    mean=0.0, stddev=stddev)

    def _glorot_initializer_conv2d(self, prev_units, num_units, mapsize, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""

        stddev  = np.sqrt(stddev_factor / (np.sqrt(prev_units*num_units)*mapsize*mapsize))
        return tf.truncated_normal([mapsize, mapsize, prev_units, num_units],
                                    mean=0.0, stddev=stddev)

    def bias_variable(self, shape, stddev=0.1):
        #TODO: stddev arg has no effect
        with tf.variable_scope(self._get_layer_str()):
            initial = tf.constant(0.1, shape=shape)
            bias = tf.get_variable('bias', initializer=initial)
            
        self.variables.append(bias)
        return bias

    def conv2d(self, x, W, stride_input = [1, 2, 2, 1]):
        return tf.nn.conv2d(x, W, strides = stride_input, padding='SAME')

    def relu(self):
        #TODO: implement leaky relu
        with tf.variable_scope(self._get_layer_str()):
            self.outputs.append(tf.nn.relu(self.get_output()))
        return self

    def full_conv2d(self, f_in, f_out, mapsize=3, stride=1, stddev_factor=0.1, weights=None):
        if weights == None:
            (weight,bias) = self.conv2d_wb_initiator(f_in, f_out, mapsize=mapsize)
        else: 
            (weight,bias) = weights
            
        self.outputs.append(self.conv2d(self.get_output(), weight, stride_input = [1, stride, stride, 1]) + bias)
        return self

    def conv2d_wb_initiator(self, f_in, f_out, mapsize=3, stddev_factor=0.1):
        return self.weight_variable([mapsize, mapsize, f_in, f_out],stddev=stddev_factor), self.bias_variable([f_out],stddev=stddev_factor)

    def batch_norm(self, scale=False):
        with tf.variable_scope(self._get_layer_str()):
            out = tf.contrib.layers.batch_norm(self.get_output(), scale=scale)
        self.outputs.append(out)
        return self

    def add_residual_block(self, f_in, f_internal, mapsize=3, num_layers=2,\
                            stddev_factor=1e-3, relu_pre_add = True):
        """Adds a residual block as per Arxiv 1512.03385, Figure 3"""

        #current = input_tensor

        #Bring size of tensor to correct size
        if(f_in != f_internal):
            self.full_conv2d( f_in, f_internal, mapsize=1, stddev_factor=1.)

        bypass = self.get_output()

        # Residual block
        for _ in range(num_layers-1):
            self.full_conv2d( f_internal, f_internal, stddev_factor=stddev_factor)
            self.batch_norm()
            self.relu()
        self.full_conv2d( f_internal, f_internal, stddev_factor=stddev_factor)
        self.batch_norm()
        if(relu_pre_add):
            self.relu()

        self.outputs.append(tf.add(self.get_output(), bypass))
        
        return self

    def dropout(self, probability):
        self.outputs.append(tf.nn.dropout(self.get_output(), probability))

    def upscale(self, size):
        self.outputs.append(tf.image.resize_bilinear(self.get_output(), size))
        return self


    def channel_concat(self, to_concat):
        tensor_1 = self.get_output()

        #TODO: Code written for TF version < 0.11. ON > 0.12, the order of parameters is changed. 
        self.outputs.append(tf.concat(3, [tensor_1, to_concat]))

    def lrelu(self, leak=.2):
        """Adds a leaky ReLU (LReLU) activation function to this model"""
        with tf.variable_scope(self._get_layer_str()):
            t1  = .5 * (1 + leak)
            t2  = .5 * (1 - leak)
            out = t1 * self.get_output() + \
                    t2 * tf.abs(self.get_output())
            
        self.outputs.append(out)
        return self

    def dense(self, prev_layer_size, n_nodes, stddev=0.1):
        shape = self.get_output().get_shape()
        assert len(shape) == 4 or len(shape) == 2, "Shape of tensor input to dense layer is not len 2 or 4" 
        if (len(shape) == 4):
                self.reshape([-1, prev_layer_size])
        
        w = self.weight_variable([prev_layer_size,n_nodes], stddev=stddev)
        b = self.bias_variable([n_nodes])
        self.outputs.append(tf.matmul(self.get_output(),w)+b)
        
    def rgb_bound(self):
        lower = tf.nn.relu(self.get_output())
        upper = tf.minimum(lower,255)
        self.outputs.append(upper)
        return self

    def image_patches(self, ksizes, strides, rates, padding = 'VALID'):
        self.outputs.append(tf.extract_image_patches(self.get_output(), ksizes, strides, rates, padding))
    

    def tanh_rgb_bound(self):
        adjusted = tf.nn.tanh(self.get_output())+1
        scaled = adjusted*255/2
        self.outputs.append(scaled)

    def reshape(self, shape):
        self.outputs.append(tf.reshape(self.get_output(),shape))
        return self



def discriminator(in_tensor):
    discriminator = Model("Discriminator", in_tensor)
    discriminator.full_conv2d(3,64, stride=1)
    discriminator.lrelu()
    
    discriminator.full_conv2d(64,128, stride=2)
    discriminator.batch_norm()
    discriminator.lrelu()
    
    discriminator.full_conv2d(128,128, stride=1)
    discriminator.batch_norm()
    discriminator.lrelu()
    
    discriminator.full_conv2d(128,192, stride=2)
    discriminator.batch_norm()
    discriminator.lrelu()
    discriminator.full_conv2d(192,192, stride=1)
    discriminator.batch_norm()
    discriminator.lrelu()
    
    discriminator.full_conv2d(192,256, stride=2)
    discriminator.batch_norm()
    discriminator.lrelu()
    discriminator.full_conv2d(256,256, stride=1)
    discriminator.batch_norm()
    discriminator.lrelu()
    
    discriminator.full_conv2d(256,512, stride=2)
    discriminator.batch_norm()
    discriminator.lrelu()
    discriminator.full_conv2d(512,512, stride=1)
    discriminator.batch_norm()
    discriminator.lrelu()
    
    discriminator.full_conv2d(512, 1024, stride=2)
    
    discriminator.dense(1024*4,1024*4)
    discriminator.lrelu()
    discriminator.dense(1024*4,1)
    return discriminator


def generator(in_tensor):
    generator = Model("Generator", in_tensor)
    
    #todo: add droput? 
    
    generator.full_conv2d(3, 64, stride=1) 
    generator.batch_norm()
    generator.lrelu()
    skip_1 = generator.get_output() #64 @ 64
    
    generator.full_conv2d(64,128, stride=2) 
    generator.batch_norm()
    generator.lrelu()
    
    generator.full_conv2d(128,256, stride=1) 
    generator.batch_norm()
    generator.lrelu()
    skip_2 = generator.get_output() # 32 @ 256  
    
    generator.full_conv2d(256,512, stride=2) 
    generator.batch_norm()
    generator.lrelu()
    
    generator.full_conv2d(512,1024, stride=1) 
    generator.batch_norm()
    generator.lrelu()
    skip_3 = generator.get_output() #16 @ 1024
    
    generator.full_conv2d(1024,1024, stride=2) # 8x8
    generator.batch_norm()
    generator.lrelu()
    
    generator.add_residual_block(1024, 1024)
    generator.add_residual_block(1024, 1024)
    generator.add_residual_block(1024, 1024)
    generator.add_residual_block(1024, 1024)
    
    generator.upscale([16,16])
    generator.channel_concat(skip_3)
    generator.full_conv2d(1024*2,512, stride=1) #16
    generator.batch_norm()
    generator.dropout(0.5)
    generator.lrelu()

    #generator.upscale([32,32])
    generator.full_conv2d(512,256, stride=1) #16
    generator.batch_norm()
    generator.dropout(0.5)
    generator.lrelu()
    
    generator.upscale([32,32])
    generator.channel_concat(skip_2)
    generator.full_conv2d(256*2,128, stride=1) 
    generator.batch_norm()
    generator.dropout(0.5)
    generator.lrelu()

    #generator.upscale([128,128])
    generator.full_conv2d(128,64, stride=1) 
    generator.batch_norm()
    generator.dropout(0.5)
    generator.lrelu()


    generator.upscale([64,64])
    generator.channel_concat(skip_1)
    generator.full_conv2d(64*2,64, stride=1) 
    generator.batch_norm()
    generator.lrelu()
    
    generator.add_residual_block(64, 64)
    generator.add_residual_block(64, 64)
    generator.full_conv2d(64,3, mapsize = 1)
    
    generator.rgb_bound()

    return generator