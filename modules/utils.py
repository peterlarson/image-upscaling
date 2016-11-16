from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc

Dimensions = namedtuple('Dimensions', ['h', 'w'])

def show_image(image):
        plt.matshow(image, cmap=plt.cm.gray)
        
#Now not random, but faster after first generation
def minibatch_image_generator(image_filenames, batch_size):
    epoch = 0 
    scrambled_images = [misc.imread(img, mode='RGB') for img in image_filenames]
    while 1:
        for batch_number in range(int(np.ceil(len(scrambled_images)/batch_size))):
            start_index = batch_number*batch_size
            end_index = min(len(scrambled_images),start_index+batch_size)
            yield epoch, scrambled_images[start_index:end_index]
        epoch = epoch + 1

def minibatch_single_image_generator(image_filenames, src_dim, out_dim):
    epoch = 0 
    scrambled_images = [misc.imread(img, mode='RGB') for img in image_filenames]
    images_to_serve = []
    if(src_dim.h // out_dim.h == 0 and src_dim.w // out_dim.w == 0):
        raise NameError("Args aren't good") 
    for index in range(len(scrambled_images)):
        curr_img = scrambled_images[index]
        if(curr_img.shape[0] != src_dim.h or curr_img.shape[1] != src_dim.w):
            scrambled_images[index] = misc.imresize(curr_img, [src_dim.h, src_dim.w], interp='cubic')
    for image in scrambled_images:
        for h_index in range(src_dim.h // out_dim.h):
            for  w_index in range(src_dim.w // out_dim.w):
                images_to_serve.append(image[h_index*out_dim.h : (h_index+1)*out_dim.h,w_index*out_dim.w : (w_index+1)*out_dim.w])
    while 1:
        for image in images_to_serve:
            yield epoch, image
        epoch = epoch + 1


def add_noise(image):
    noise = np.random.normal(0, 10, image.shape[0]*image.shape[1]*image.shape[2])
    noise_reshaped = np.reshape(noise, image.shape) + image
    to_return = np.clip(noise_reshaped,0,255)
    return to_return