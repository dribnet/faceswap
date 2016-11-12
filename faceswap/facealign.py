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
import faceswap.core

import sys

# default extension is 40% of shortest edge. can be clipped further
#   in pixels by setting max_extension_amount
# if longest edge is longer than max_input_image_extent, it will be
#   scaled down to max_input_image_extent
def read_im_and_landmarks(fname, max_extension_amount=-1, max_input_image_extent=2048):
    blur_amount = 31
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    if (im is None):
        raise faceswap.core.NoFaces

    # max_input_image_extent = 2048

    x_scale_factor = float(max_input_image_extent) / im.shape[0]
    y_scale_factor = float(max_input_image_extent) / im.shape[1]
    scale_factor = min(1.0, x_scale_factor, y_scale_factor)

    if scale_factor < 1.0:
        im_core = cv2.resize(im, (int(im.shape[1] * scale_factor),
                                  int(im.shape[0] * scale_factor)))
    else:
        im_core = im

    core_shape = im_core.shape
    # add a fuzzy buffer the same width all the way around
    min_dimension = core_shape[0]
    if core_shape[1] < min_dimension:
        min_dimension = core_shape[1]
    extension_amount = 0.4 * min_dimension
    if max_extension_amount >=0 and extension_amount > max_extension_amount:
        extension_amount = max_extension_amount
    left   = int(core_shape[1] - extension_amount)
    right  = int(2 * core_shape[1] + extension_amount)
    top    = int(core_shape[0] - extension_amount)
    bottom = int(2 * core_shape[0] + extension_amount)
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
    rects, landmarks = faceswap.core.get_landmarks(im_final, extension_amount)
    # cv2.imwrite("debug.png", im_final)

    return im_final, rects, landmarks


if __name__ == "__main__":
    avg_landmarks = np.load("mean_landmark_x4.npy")
    im, landmarks = faceswap.core.read_im_and_landmarks("celeba/000001.jpg")
    coerced_landmarks = 0 * landmarks + 4 * avg_landmarks

    source_dir = "/Volumes/expand1/develop/data/CelebA/original/img_celeba"
    dest_dir = "/Volumes/expand1/develop/data/CelebA/original/dlib_aligned_256"

    num_images = 202599
    # num_images = 10
    for i in range(num_images):
    # for i in range(14156,50000):
        try:
            filebase = "{:06d}".format(i+1)
            if i % 10000 == 0:
                print("face {}".format(filebase))
            im, rects, landmarks = read_im_and_landmarks("{}/{}.jpg".format(source_dir, filebase))
            M = faceswap.core.transformation_from_points(coerced_landmarks[faceswap.core.ALIGN_POINTS],
                                           landmarks[faceswap.core.ALIGN_POINTS])
            warped_im2 = faceswap.core.warp_im(im, M, (1024,1024,3))
            resize64 = imresize(warped_im2, (256,256), interp="bicubic", mode="RGB")
            cv2.imwrite("{}/{}.png".format(dest_dir, filebase), resize64)
            # cv2.imwrite("{}/{}.png".format(dest_dir, filebase), im)
        except faceswap.core.NoFaces:
            pass
        except faceswap.core.TooManyFaces:
            print("too many faces in {}".format(filebase))
        # except:
        #     print "Unexpected error:", sys.exc_info()[0]
