#!/usr/bin/python

# Copyright (c) 2015 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This is the code behind the Switching Eds blog post:

    http://matthewearl.github.io/2015/07/28/switching-eds-with-python/

See the above for an explanation of the code below.

To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV. You'll also need to obtain the trained model from
sourceforge:

    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

Unzip with `bunzip2` and change `PREDICTOR_PATH` to refer to this file. The
script is run like so:

    ./faceswap.py <head image> <face image>

If successful, a file `output.jpg` will be produced with the facial features
from `<head image>` replaced with the facial features from `<face image>`.

"""

import cv2
import dlib
import numpy as np
from scipy.misc import imresize 
import faceswap

import sys

def read_im_and_landmarks(fname):
    blur_amount = 31
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    if (im is None):
        raise faceswap.NoFaces
    im_core = cv2.resize(im, (im.shape[1] * faceswap.SCALE_FACTOR,
                              im.shape[0] * faceswap.SCALE_FACTOR))

    core_shape = im_core.shape
    left   = int(core_shape[1] - 0.25 * core_shape[1])
    right  = int(2 * core_shape[1] + 0.25 * core_shape[1])
    top    = int(core_shape[0] - 0.25 * core_shape[0])
    bottom = int(2 * core_shape[0] + 0.25 * core_shape[0])
    # print("L,R,T,B {},{},{},{}".format(left, right, top, bottom))

    im_blur = cv2.GaussianBlur(im_core, (blur_amount, blur_amount), 0)
    im_blur = cv2.GaussianBlur(im_blur, (blur_amount, blur_amount), 0)
    blur_flipx = cv2.flip(im_blur, 1)
    blur_flipy = cv2.flip(im_blur, 0)
    blur_flipxy = cv2.flip(im_blur, -1)
    im_row1 = np.concatenate((blur_flipxy, blur_flipy, blur_flipxy), axis=1)
    im_row2 = np.concatenate((blur_flipx, im_core, blur_flipx), axis=1)
    im_row3 = np.concatenate((blur_flipxy, blur_flipy, blur_flipxy), axis=1)
    im_buffered = np.concatenate((im_row1, im_row2, im_row3), axis=0)
    im_final = im_buffered[top:bottom, left:right, :].astype(np.uint8)
    s = faceswap.get_landmarks(im_final)

    return im_final, s


if __name__ == "__main__":
    avg_landmarks = np.load("mean_landmark_x4.npy")
    im, landmarks = faceswap.read_im_and_landmarks("celeba/000001.jpg")
    coerced_landmarks = 0 * landmarks + avg_landmarks

    source_dir = "/Volumes/expand1/develop/data/CelebA/original/img_celeba"
    dest_dir = "/Volumes/expand1/develop/data/CelebA/original/dlib_aligned_128"

    num_images = 202599
    # num_images = 10
    for i in range(num_images):
        try:
            filebase = "{:06d}".format(i+1)
            if i % 10000 == 0:
                print("face {}".format(filebase))
            im, landmarks = read_im_and_landmarks("{}/{}.jpg".format(source_dir, filebase))
            M = faceswap.transformation_from_points(coerced_landmarks[faceswap.ALIGN_POINTS],
                                           landmarks[faceswap.ALIGN_POINTS])
            warped_im2 = faceswap.warp_im(im, M, (256,256,3))
            resize64 = imresize(warped_im2, (128,128), interp="bicubic", mode="RGB")
            cv2.imwrite("{}/{}.png".format(dest_dir, filebase), resize64)
            # cv2.imwrite("{}/{}.png".format(dest_dir, filebase), im)
        except faceswap.NoFaces:
            pass
        except faceswap.TooManyFaces:
            print("too many faces in {}".format(filebase))
        # except:
        #     print "Unexpected error:", sys.exc_info()[0]
