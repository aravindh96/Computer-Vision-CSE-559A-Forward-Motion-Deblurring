from skimage.io import imread, imsave
import numpy as np
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
from scipy.ndimage import gaussian_filter
from numpy.linalg import matrix_rank
import cv2


warnings.filterwarnings('ignore')

im = np.float32(imread(fn('dot_input.jpg')))
normal = np.array([[0],[0],[1]])
print(normal.shape,"hello")

#Sampling translation along z axis
tz = (np.arange(0,1,5))
# print(tz[3])
tx = 3
ty = 4
T = np.zeros((3,3,5))
H = np.zeros((3,3,5))
rows,cols,ch = im.shape
# Translational matrix
for i in range(np.size(tz)):
    # print(i)

    #
    T[:,:,i] =[[1+tx*normal[0], tx*normal[1], tx*normal[2]],
        [ty*normal[0],1+ty*normal[1],ty*normal[2]],
        [tz[i]*normal[0],tz[i]*normal[1],1+tz[i]*normal[2]]]

    # T[:,:,i] =[[1+1*normal[0], 1*normal[1], 1*normal[2]],
    #     [2*normal[0],1+2*normal[1],2*normal[2]],
    #     [tz[i]*normal[0],tz[i]*normal[1],1+tz[i]*normal[2]]]

# print(T[:,:,3])

#Camera matrix
f = 674.91
K = np.array([[f,0,600],
    [0,f,600],
    [0,0,1]])

# print(np.linalg.inv(K))

for i in range(np.size(tz)):
    H[:,:,i] = K*T[:,:,i]*np.linalg.inv(K)

# M = cv2.getPerspectiveTransform(pts1,pts2)
h = H[:,:,3].astype(np.float64)
print(h)

filter = cv2.warpPerspective(im,h,(cols,rows))
# print(np.shape(im))

# test = np.ones((4,4))
# # print(test)
# sigma =[0.1,0.5]
#
# filter = gaussian_filter(im, sigma = 0.5)


imsave(fn("filter.jpg"),filter)
