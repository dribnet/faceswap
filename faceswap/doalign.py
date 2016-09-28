#!/usr/bin/python


"""
Use dlib to align a face to standard landmarks.

Can run on all files in a directory or a single file.
"""

import cv2
import dlib
import numpy as np
from scipy.misc import imresize 

import glob
import os
import sys
import faceswap.core
from faceswap import facealign
import argparse

# get a matrix which represents the standard face location (1024x1024 I believe)
def get_standard_landmarks():
    # historical note: this is how we made this file
    # avg_landmarks = np.load("mean_landmark_x4.npy")
    # im, landmarks = facealign.read_im_and_landmarks("celeba/000001.jpg")
    # im, landmarks = faceswap.core.read_im_and_landmarks("celeba/000001.jpg")
    # standard_landmarks = 0 * landmarks + 4 * avg_landmarks
    # np.save("standard_landmarks", standard_landmarks)

    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = "standard_landmarks.npy"
    abs_file_path = os.path.join(script_dir, rel_path)
    return np.matrix(np.load(abs_file_path))

# alignment from infile saved to outfile. returns true if all is ok.
def align_face(infile, outfile, image_size, standard_landmarks=None, exception_print=True, max_extension_amount=-1):
    if standard_landmarks is None:
        standard_landmarks = get_standard_landmarks()
    try:
        im, landmarks = facealign.read_im_and_landmarks(infile, max_extension_amount=max_extension_amount)
        M = faceswap.core.transformation_from_points(standard_landmarks[faceswap.core.ALIGN_POINTS],
                                       landmarks[faceswap.core.ALIGN_POINTS])
        warped_im2 = faceswap.core.warp_im(im, M, (1024,1024,3))
        resize64 = imresize(warped_im2, (image_size, image_size), interp="bicubic", mode="RGB")
        cv2.imwrite(outfile, resize64)
        return True
    except faceswap.core.NoFaces:
        if exception_print:
            print("no faces in {}".format(infile))
        return False
    except faceswap.core.TooManyFaces:
        if exception_print:
            print("too many faces in {}".format(infile))
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align faces")
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of output images")
    parser.add_argument("--max-extension-amount", dest='max_extension_amount', type=int, default=-1,
                        help="maximum pixels to extend (0 to disable, -1 to ignore)")
    parser.add_argument("--input-directory", dest='input_directory', default="inputs",
                        help="directory for input files")
    parser.add_argument("--output-directory", dest='output_directory', default="outputs",
                        help="directory for output files")
    parser.add_argument("--input-file", dest='input_file', default=None,
                        help="single file input (overrides input-directory)")
    parser.add_argument("--output-file", dest='output_file', default="output.png",
                        help="single file output")
    args = parser.parse_args()

    landmarks = get_standard_landmarks()

    if args.input_file is not None:
        if align_face(args.input_file, args.output_file, args.image_size, landmarks, max_extension_amount=args.max_extension_amount):
            sys.exit(0)
        else:
            sys.exit(1)

    # read input files
    files = glob.glob("{}/*.*".format(args.input_directory))
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    for infile in files:
        outfile = os.path.join(args.output_directory, os.path.basename(infile))
        # always save as png
        outfile = "{}.png".format(os.path.splitext(outfile)[0])
        align_face(infile, outfile, args.image_size, landmarks)
        # except:
        #     print "Unexpected error:", sys.exc_info()[0]
