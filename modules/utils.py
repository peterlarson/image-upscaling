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

def showImagesHorizontally(list_of_files):
    fig = plt.figure(figsize=(16, 12))
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        plt.imshow(list_of_files[i])
        plt.axis('off')

def save_output(image, name, number):
    misc.imsave("data/generated_output/"+name+_"+str(number)+".png", image)