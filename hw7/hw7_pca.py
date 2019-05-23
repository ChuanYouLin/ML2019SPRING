import skimage.io
import numpy as np
import sys
import os
from os import listdir

image_names = listdir(sys.argv[1])
image_X = []

for name in image_names:
    single_img = skimage.io.imread('./' + os.path.join(sys.argv[1],name))
    image_X.append(single_img)
image_flat = np.reshape(image_X,(415,-1))
mean_face = np.mean(image_flat,axis=0)

image_center = image_flat - mean_face

U, S, V = np.linalg.svd(image_center.T, full_matrices=False) 

input_img = skimage.io.imread(os.path.join(sys.argv[1],sys.argv[2])).flatten()
input_img_center = input_img - mean_face

weights = np.dot(input_img_center, U[:, :5])

recon = mean_face + np.dot(weights, U[:, :5].T)
recon -= np.min(recon)
recon /= np.max(recon)
recon = (recon * 255).astype(np.uint8)


skimage.io.imsave(sys.argv[3], recon.reshape(600,600,3))