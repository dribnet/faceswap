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
import numpy

import sys
import os

# PREDICTOR_PATH = "/home/matt/dlib-18.16/shape_predictor_68_face_landmarks.dat"
# PREDICTOR_PATH = "/Users/tom/code/nets/faceswap/shape_predictor_68_face_landmarks.dat"
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
PREDICTOR_PATH = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass


def get_bounding_box(landmarks):
    min_x, min_y = numpy.asarray(numpy.min(landmarks, axis=0)).reshape(-1)
    max_x, max_y = numpy.asarray(numpy.max(landmarks, axis=0)).reshape(-1)
    return min_x, max_x, min_y, max_y

def get_extent(landmarks):
    min_x, max_x, min_y, max_y = get_bounding_box(landmarks)
    return max_x - min_x, max_y - min_y

def get_max_extent(landmarks):
    extent_x, extent_y = get_extent(landmarks)
    return numpy.max([extent_x, extent_y])

# when ordering landmarks, choose largest
def landmark_ordering(landmark_pair):
    landmarks = landmark_pair[1]
    min_x, min_y = numpy.asarray(numpy.min(landmarks, axis=0)).reshape(-1)
    max_extent = get_max_extent(landmarks)
    return [max_extent, min_x, min_y]

def rect_not_in_border(r, image_width, image_height, border_filter_width):
    min_x, max_x, min_y, max_y = get_bounding_box(r[1])
    if min_x < border_filter_width or min_y < border_filter_width or \
       max_x > image_width - border_filter_width or max_y > image_height - border_filter_width:
        return False
    return True

def get_landmarks(im, border_filter_width=0):
    rects = detector(im, 1)

    # compute matrix for each rect
    pairs = map(lambda rect: [rect, numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])], rects)

    image_height, image_width, image_depth = im.shape
    pairs = filter(lambda rect: rect_not_in_border(rect, image_width, image_height, border_filter_width), pairs)


    if len(rects) == 0:
        raise NoFaces

    if len(pairs) > 1:
        # print(get_bounding_box(pairs[0][1]))
        # print(get_bounding_box(pairs[1][1]))
        new_pairs = sorted(pairs, key=landmark_ordering, reverse=True)
        pairs = [ new_pairs[0] ]
        # print(get_bounding_box(pairs[0][1]))
        # raise TooManyFaces

    return pairs[0][1]

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im
    
def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))

def do_faceswap_from_saved(body_im, body_landmarks, face_im, face_landmarks, output_image):
    M = transformation_from_points(body_landmarks[ALIGN_POINTS],
                                   face_landmarks[ALIGN_POINTS])

    mask = get_face_mask(face_im, face_landmarks)
    warped_mask = warp_im(mask, M, body_im.shape)
    combined_mask = numpy.max([get_face_mask(body_im, body_landmarks), warped_mask],
                              axis=0)

    warped_im2 = warp_im(face_im, M, body_im.shape)
    warped_corrected_im2 = correct_colours(body_im, warped_im2, body_landmarks)

    output_im = body_im * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

    cv2.imwrite(output_image, output_im)

def do_faceswap_from_face(body_image, face_im, face_landmarks, output_image):
    body_im, body_landmarks = read_im_and_landmarks(body_image)
    return do_faceswap_from_saved(body_im, body_landmarks, face_im, face_landmarks, output_image)

def do_faceswap(body_image, face_image, output_image):
    face_im, face_landmarks = read_im_and_landmarks(face_image)
    do_faceswap_from_face(body_image, face_im, face_landmarks, output_image)

if __name__ == "__main__":
    do_faceswap(sys.argv[1], sys.argv[2], sys.argv[3])
