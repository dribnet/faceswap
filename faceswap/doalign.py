#!/usr/bin/python


"""
Use dlib to align a face to standard landmarks.

Can run on all files in a directory or a single file.
"""

import cv2
import dlib
import numpy as np
from scipy.misc import imresize 

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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

# alignment from infile saved to outfile. returns true, rect if all is ok.
# if detected face is not width min_span, returns False
def align_face(infile, outfile, image_size, standard_landmarks=None, min_span=None, exception_print=True, max_extension_amount=-1):
    if standard_landmarks is None:
        standard_landmarks = get_standard_landmarks()
    try:
        if min_span is not None:
            max_input_image_extent = 8 * min_span
        else:
            max_input_image_extent = 2048
        im, rect, landmarks = facealign.read_im_and_landmarks(infile, max_extension_amount=max_extension_amount, max_input_image_extent=max_input_image_extent)
        if min_span is not None:
            width = rect.right()-rect.left()
            height = rect.bottom()-rect.top()
            if width < min_span or height < min_span:
                return False, rect
        M = faceswap.core.transformation_from_points(standard_landmarks[faceswap.core.ALIGN_POINTS],
                                       landmarks[faceswap.core.ALIGN_POINTS])
        warped_im2 = faceswap.core.warp_im(im, M, (1024,1024,3))
        resize64 = imresize(warped_im2, (image_size, image_size), interp="bicubic", mode="RGB")
        cv2.imwrite(outfile, resize64)
        return True, rect
    except faceswap.core.NoFaces:
        if exception_print:
            print("no faces in {}".format(infile))
        return False, None
    except faceswap.core.TooManyFaces:
        if exception_print:
            print("too many faces in {}".format(infile))
        return False, None

# TODO: fast hack of above. refactorme
def align_face_buffer(im_buf, image_size, standard_landmarks=None, min_span=None, exception_print=True, max_extension_amount=-1):
    if standard_landmarks is None:
        standard_landmarks = get_standard_landmarks()
    try:
        if min_span is not None:
            max_input_image_extent = 8 * min_span
        else:
            max_input_image_extent = 2048
        im, rect, landmarks = facealign.read_im_and_landmarks("", max_extension_amount=max_extension_amount, max_input_image_extent=max_input_image_extent, im_buf=im_buf)
        if min_span is not None:
            width = rect.right()-rect.left()
            height = rect.bottom()-rect.top()
            if width < min_span or height < min_span:
                return False, None, rect
        M = faceswap.core.transformation_from_points(standard_landmarks[faceswap.core.ALIGN_POINTS],
                                       landmarks[faceswap.core.ALIGN_POINTS])
        warped_im2 = faceswap.core.warp_im(im, M, (1024,1024,3))
        resize64 = imresize(warped_im2, (image_size, image_size), interp="bicubic", mode="RGB")
        return True, resize64, rect
    except faceswap.core.NoFaces:
        if exception_print:
            print("no faces")
        return False, None, None
    except faceswap.core.TooManyFaces:
        if exception_print:
            print("too many faces")
        return False, None, None

class NewFileHandler(FileSystemEventHandler):
    def setup(self, outdir, image_size, landmarks, min_span, max_extension_amount):
        self.outdir = outdir
        self.image_size = image_size
        self.landmarks = landmarks
        self.min_span = min_span
        self.max_extension_amount = max_extension_amount

    def process(self, infile):
        print("Processing file: {}".format(infile))
        outfile = os.path.join(args.output_directory, os.path.basename(infile))
        # always save as png
        outfile = "{}.png".format(os.path.splitext(outfile)[0])
        print("Processing {} to {}".format(infile, outfile))
        align_face(infile, outfile, self.image_size, self.landmarks, self.min_span, max_extension_amount=self.max_extension_amount)

    def on_modified(self, event):
        if not event.is_directory:
            self.process(event.src_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align faces")
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of output images")
    parser.add_argument("--min-span", dest='min_span', type=int, default=None,
                        help="drop images if w/h of detected face is lower than span")
    parser.add_argument("--max-extension-amount", dest='max_extension_amount', type=int, default=-1,
                        help="maximum pixels to extend (0 to disable, -1 to ignore)")
    parser.add_argument("--input-directory", dest='input_directory', default="inputs",
                        help="directory for input files")
    parser.add_argument("--output-directory", dest='output_directory', default="outputs",
                        help="directory for output files")
    parser.add_argument('--watch', dest='watch', default=False, action='store_true',
                        help="monitor input-directory indefinitely")
    parser.add_argument("--input-file", dest='input_file', default=None,
                        help="single file input (overrides input-directory)")
    parser.add_argument("--output-file", dest='output_file', default="output.png",
                        help="single file output")
    args = parser.parse_args()

    landmarks = get_standard_landmarks()

    if args.input_file is not None:
        did_align, rect = align_face(args.input_file, args.output_file, args.image_size, landmarks, args.min_span, max_extension_amount=args.max_extension_amount)
        if did_align:
            sys.exit(0)
        else:
            sys.exit(1)

    # read input files
    files = sorted(glob.glob("{}/*.*".format(args.input_directory)))
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    event_handler = NewFileHandler()
    event_handler.setup(args.output_directory, args.image_size, landmarks, args.min_span, args.max_extension_amount)

    for f in files:
        event_handler.process(f)

    if args.watch:
        print("Watching input directory {}".format(args.input_directory))
        observer = Observer()
        observer.schedule(event_handler, path=args.input_directory, recursive=False)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
