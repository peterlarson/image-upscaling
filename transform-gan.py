
'''
This code was developed in an Anaconda Distribution of Python 3.5

'''
import numpy as np
from scipy import ndimage
import tensorflow as tf
from os import listdir

#has package pillow as a dependency. Is needed to give scipy image operations. 
from scipy import misc
import scipy

from modules.utils import *
from modules.model import *
import matplotlib.pyplot as plt
import modules.vgg19 as vgg19

vgg_r = vgg19.Vgg19(vgg19_npy_path="data/vgg19.npy")
vgg_f = vgg19.Vgg19(vgg19_npy_path="data/vgg19.npy")


'''
Constants
'''
IMAGE_FOLDER_c1 = "data/64_celebA"
IMAGE_FOLDER_c2 = "data/64_celebA_grey"

SAMPLE_FOLDER = "data/generated_output"
SNAPSHOT_INTERVAL = 300
TOTAL_STEPS = 6000


BATCH_SIZE = 10
EPOCHS = 35

INPUT_SIZE = Dimensions(64, 64)

GEN_RATE = 0.001
DISC_RATE = 0.0005


DISC_REAL_COEF = 1
DISC_FAKE_COEF = 1
GEN_VGG_COEF = 0.0001
GEN_DISC_COEF = 1


'''Load data'''
image_names_c1 = listdir(IMAGE_FOLDER_c1)
image_names_c2 = listdir(IMAGE_FOLDER_c2)
image_paths_c1 = [IMAGE_FOLDER_c1 + '/' + name for name in image_names_c1[0:10000]]
image_paths_c2 = [IMAGE_FOLDER_c2 +'/' + name for name in image_names_c2[0:10000]]


'''
Tensorflow Model Construction
'''
disc_real_input = tf.placeholder('float32', shape = [BATCH_SIZE, INPUT_SIZE.h, INPUT_SIZE.w, 3])
gen_input = tf.placeholder('float32',shape = [BATCH_SIZE, INPUT_SIZE.h,INPUT_SIZE.w,3])

#Generator
generator_model = generator(gen_input)
gen_output = generator_model.outputs[-2]


#Discriminator
with tf.variable_scope('disc') as scope:
    disc_model_r = discriminator(disc_real_input)
    scope.reuse_variables()
    disc_model_f = discriminator(gen_output)

    
#fake_model = patch_discriminator(disc_real_input)
    
#vgg
#with tf.variable_scope('vgg') as scope: 
vgg_r.build(gen_input, dimension = INPUT_SIZE.h)
    #scope.reuse_variables()
vgg_f.build(generator_model.get_output(), dimension = INPUT_SIZE.h)
    
    
#discriminator loss
disc_c1 = tf.nn.sigmoid_cross_entropy_with_logits(disc_model_r.get_output(), tf.ones_like(disc_model_r.get_output()))
disc_c2 = tf.nn.sigmoid_cross_entropy_with_logits(disc_model_f.get_output(), tf.zeros_like(disc_model_f.get_output()))

disc_loss = DISC_REAL_COEF*tf.reduce_mean(disc_c1)+DISC_FAKE_COEF*tf.reduce_mean(disc_c2)
disc_opt = tf.train.AdamOptimizer(learning_rate=DISC_RATE).minimize(disc_loss, var_list = disc_model_f.variables)

#Generator Loss

#vgg loss
diff = tf.reshape(vgg_r.conv4_4, [BATCH_SIZE,-1]) - tf.reshape(vgg_f.conv4_4, [BATCH_SIZE,-1])
vgg_mse = tf.reduce_mean(tf.abs(diff))*GEN_VGG_COEF

#disc loss
gen_ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_model_f.get_output(), tf.ones_like(disc_model_f.get_output())))

#combined
gen_loss = GEN_DISC_COEF*gen_ce + vgg_mse
gen_opt = tf.train.AdamOptimizer(learning_rate=GEN_RATE).minimize(gen_loss,var_list = generator_model.variables)
gen_img = tf.cast(generator_model.get_output(), dtype=tf.uint8)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

'''
Loding Data
'''
image_gen_c1 = minibatch_image_generator(image_paths_c1[0:(len(image_paths_c1)//BATCH_SIZE)*BATCH_SIZE],BATCH_SIZE)
image_gen_c2 = minibatch_image_generator(image_paths_c2[0:(len(image_paths_c2)//BATCH_SIZE)*BATCH_SIZE],BATCH_SIZE)
print("generating images") 
_,image_batch_c1 = next(image_gen_c1)
_,image_batch_c2 = next(image_gen_c2)
print("generated images") 
#Setting up samples
_,image_batch_c1 = next(image_gen_c1)
_,image_batch_c2 = next(image_gen_c2)
fd_sample = {disc_real_input:image_batch_c1, gen_input:image_batch_c2}



'''
Training Loop
'''

try:
    index = 1
    steps = TOTAL_STEPS
    while(index < steps):
        index = index + 1
        _,image_batch_c1 = next(image_gen_c1)
        _,image_batch_c2 = next(image_gen_c2)
        
        fd = {disc_real_input:image_batch_c1, gen_input:image_batch_c2}
        sess.run(disc_opt, feed_dict = fd)
        
        _,image_batch_c2 = next(image_gen_c2)
        fd = {gen_input:image_batch_c2}
        sess.run(gen_opt, feed_dict = fd)
        
        _,image_batch_c2 = next(image_gen_c2)
        fd = {gen_input:image_batch_c2}
        sess.run(gen_opt, feed_dict = fd)
        
        if(index % SNAPSHOT_INTERVAL == 0):        
            _,image_batch_c1 = next(image_gen_c1)
            _,image_batch_c2 = next(image_gen_c2)
            fd = {disc_real_input:image_batch_c1, gen_input:image_batch_c2}
            (g_comb_error, g_vgg_error, g_disc_error, d_error) = sess.run([gen_loss, vgg_mse, gen_ce, disc_loss], feed_dict = fd)
            print('step {}, gcom_loss {:01.2f}, gvgg_e {:01.2f}, gdisc_e {:01.2f}, d_loss {:01.2f}'.format(index,g_comb_error, g_vgg_error, g_disc_error, d_error))
            (sample) = sess.run(gen_img,feed_dict=fd_sample)
except KeyboardInterrupt:
    pass #avoids getting KeyboardInterrupt errors when stopping training early
except:



'''
Sample Generation
'''
epoch,image_batch_c1 = next(image_gen_c1)
epoch,image_batch_c2 = next(image_gen_c2)
fd_sample = {disc_real_input:image_batch_c1, gen_input:image_batch_c2}
(sample) = sess.run(gen_img,feed_dict=fd_sample)
for i in range(BATCH_SIZE):
    misc.imsave(SAMPLE_FOLDER +"/input_"+str(i)+".png", image_batch_c2[i])
    misc.imsave(SAMPLE_FOLDER +"/output_"+str(i)+".png", sample[i])
