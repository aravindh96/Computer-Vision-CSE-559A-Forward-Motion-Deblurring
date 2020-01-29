## Default modules imported. Import more if you need to.

import numpy as np
import math
import cv2

# Splices the source image into a quadrilateral in the dest image,
# where dpts in a 4x2 image with each row giving the [x,y] co-ordinates
# of the corner points of the quadrilater (in order, top left, top right,
# bottom left, and bottom right).
#
# Note that both src and dest are color images.
#
# Return a spliced color image.
def get_grid(x, y, homogenous=False):
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1]))) if homogenous else coords

def splice(src):

    img = cv2.imread('dot_input.jpg')
    rows,cols,ch = img.shape
    pts1 = np.float32([[5,5],[295,5],[5,295],[295,295]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    print(M)
    # h = np.array([[ 1,0,0],[ 1,0,0],[ 0,0,1]]).astype(np.float64)
    h = np.array([[ 1,0,50],
    [ 0, 1,50],
    [ 0, 0, 1.1]]).astype(np.float64)

    h2 = np.array([[ 1,0,50],
    [ 0, 1,50],
    [ 0, 0, 1.4]]).astype(np.float64)

    h3 = np.array([[ 1,0,50],
    [ 0, 1,50],
    [ 0, 0, 1.5]]).astype(np.float64)

    h1 = np.array([[ 1,0,0],
    [ 0, 1,0],
    [ 0, 0, 1]]).astype(np.float64)

    print(M.dtype,h.dtype)
    dest1 = cv2.warpPerspective(img,h,(cols,rows))
    dest2 = cv2.warpPerspective(img,h1,(cols,rows))
    dest3 = cv2.warpPerspective(img,h2,(cols,rows))
    dest4 = cv2.warpPerspective(img,h3,(cols,rows))

    dest = 0.25*dest1 + 0.25*dest2 +0.25*dest3 + 0.25*dest4
    # plt.subplot(121),plt.imshow(img),plt.title('Input')
    # plt.subplot(122),plt.imshow(dst),plt.title('Output')
    # plt.show()

    # H,W = np.shape(src)[0:2]
    # dest = np.zeros_like(src)
    #
    # print(src.shape)
    #
    # dest = cv2.warpPerspective(src,h,(H,W))


    # image = src.copy()
    # height, width = image.shape[:2]
    # tx, ty = np.array((width // 2, height // 2))
    # angle = np.radians(45)
    # scale = 2.0
    # R = np.array([
    #     [np.cos(angle), np.sin(angle), 0],
    #     [-np.sin(angle), np.cos(angle), 0],
    #     [0, 0, 1]
    # ])
    # T = np.array([
    #     [1, 0, tx],
    #     [0, 1, ty],
    #     [0, 0, 1]
    # ])
    # S = np.array([
    #     [scale, 0, 0],
    #     [0, scale, 0],
    #     [0, 0, 1]
    # ])
    # A = T @ R @ S @ np.linalg.inv(T)



    # # Grid to represent image coordinate
    # coords = get_grid(width, height, True)
    # x_ori, y_ori = coords[0], coords[1]
    # # Apply transformation
    # warp_coords = np.round(A@coords).astype(np.int)
    # xcoord2, ycoord2 = warp_coords[0, :], warp_coords[1, :]
    # # Get pixels within image boundary
    # indices = np.where((xcoord2 >= 0) & (xcoord2 < width) &
    #                    (ycoord2 >= 0) & (ycoord2 < height))
    # xpix2, ypix2 = xcoord2[indices], ycoord2[indices]
    # xpix, ypix = x_ori[indices], y_ori[indices]
    # # Map the pixel RGB data to new location in another array
    # canvas = np.zeros_like(image)
    # canvas[int(ypix2), int(xpix2)] = image[int(ypix), int(xpix)]

    return dest


########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


simg = np.float32(imread(fn('dot_input.jpg')))/255.
dimg = np.float32(imread(fn('test1.jpg')))/255.
dpts = np.float32([ [276,54],[406,79],[280,182],[408,196]]) # Hard coded

comb = splice(simg)
# pts = np.array([[1,1,1,1],[2,1,4,1],[1,1,3,3],[1,2,1,5]])
# getH(pts)

imsave(fn('test.jpg'),comb)
