import faceswap.core

import cv2
import dlib
import numpy

import sys
import os
import glob
import argparse
import tempfile

global_grid_size = 128

# when ordering landmarks, choose largest
def page_ordering(landmark_pair):
    landmarks = landmark_pair[1]
    min_x, min_y = numpy.asarray(numpy.min(landmarks, axis=0)).reshape(-1)
    min_x = int(min_x / global_grid_size)
    min_y = int(min_y / global_grid_size)
    return [min_x, min_y]

def multi_get_landmarks(im, border_filter_width=0):
    rects = faceswap.core.detector(im, 1)

    if len(rects) == 0:
        raise NoFaces

    # compute matrix for each rect
    pairs = map(lambda rect: [rect, numpy.matrix([[p.x, p.y] for p in faceswap.core.predictor(im, rect).parts()])], rects)

    image_height, image_width, image_depth = im.shape
    pairs = filter(lambda rect: faceswap.core.rect_not_in_border(rect, image_width, image_height, border_filter_width), pairs)

    if len(pairs) == 0:
        raise NoFaces

    sorted_pairs = sorted(pairs, key=page_ordering, reverse=False)
    sorted_pairs = [x[1] for x in sorted_pairs]

    return sorted_pairs

def multi_read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * faceswap.core.SCALE_FACTOR,
                         im.shape[0] * faceswap.core.SCALE_FACTOR))
    s = multi_get_landmarks(im)

    return im, s

def perform_faceswap_from_saved(body_im, body_landmarks, face_im, face_landmarks, tight_mask=False):
    M = faceswap.core.transformation_from_points(body_landmarks[faceswap.core.ALIGN_POINTS],
                                   face_landmarks[faceswap.core.ALIGN_POINTS])

    mask = faceswap.core.get_face_mask(face_im, face_landmarks, tight_mask)
    warped_mask = faceswap.core.warp_im(mask, M, body_im.shape)
    combined_mask = numpy.max([faceswap.core.get_face_mask(body_im, body_landmarks, tight_mask), warped_mask],
                              axis=0)

    warped_im2 = faceswap.core.warp_im(face_im, M, body_im.shape)
    # warped_corrected_im2 = correct_colours(body_im, warped_im2, body_landmarks)

    output_im = body_im * (1.0 - combined_mask) + warped_im2 * combined_mask

    return output_im

def multi_do_faceswap_from_face(body_image, face_im, face_landmarks_list, output_image, tight_mask=False):
    body_im, body_landmarks_list = multi_read_im_and_landmarks(body_image)
    output_im = body_im
    if len(body_landmarks_list) != len(face_landmarks_list):
        print("Warning: putting {} faces on image with {} faces".format(
            len(face_landmarks_list), len(body_landmarks_list)))
    # print("Replacing {} faces".format(len(body_landmarks_list)))
    # print("Found {} faces".format(len(face_landmarks_list)))
    for i in range(len(body_landmarks_list)):
        output_im = perform_faceswap_from_saved(output_im, body_landmarks_list[i], face_im, face_landmarks_list[i], tight_mask)
    cv2.imwrite(output_image, output_im)

def do_faceswap(body_image, face_image, output_image, tight_mask=False):
    face_im, face_landmarks_list = multi_read_im_and_landmarks(face_image)
    multi_do_faceswap_from_face(body_image, face_im, face_landmarks_list, output_image, tight_mask)

def strip_to_base(infile):
    tdir = tempfile.gettempdir()
    base_file1 = os.path.join(tdir, "base1.png")
    base_file = os.path.join(tdir, "base.png")
    os.system("convert {0} -crop {1}x{1}+0+0 {2}".format(args.input_file, args.image_size, base_file1))
    os.system("convert {0} {0} {0} {0} {0} {0} {0} {0} +append {1}".format(base_file1, base_file))
    return base_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align faces")
    parser.add_argument("--image-size", dest='image_size', type=int, default=128,
                        help="size of output images")
    parser.add_argument("--max-extension-amount", dest='max_extension_amount', type=int, default=-1,
                        help="maximum pixels to extend (0 to disable, -1 to ignore)")
    parser.add_argument("--input-glob", dest='input_glob', default="inputs",
                        help="glob for input files")
    parser.add_argument("--output-directory", dest='output_directory', default="outputs",
                        help="directory for output files")
    parser.add_argument("--base-file", dest='base_file', default=None,
                        help="main file to use as ground for faceswap")
    parser.add_argument("--input-file", dest='input_file', default=None,
                        help="single file input (overrides input-directory)")
    parser.add_argument("--output-file", dest='output_file', default="output.png",
                        help="single file output")
    parser.add_argument("--strip", dest='strip', action='store_true', default=False,
                        help="Base-file is top left of image-file")
    parser.add_argument("--tight-mask", dest='tight_mask', action='store_true', default=False,
                        help="Use a tight mask around facial features")
    args = parser.parse_args()

    global_grid_size = args.image_size

    if args.input_file is not None:
        if args.strip:
            base_file = strip_to_base(args.input_file)
        else:
            base_file = args.base_file
        do_faceswap(base_file, args.input_file, args.output_file, args.tight_mask)
        sys.exit(0)

    # instead, use input-directory and output-directory
    files = sorted(glob.glob(args.input_glob))
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    print("Processing {} files".format(len(files)))
    for infile in files:
        outfile = os.path.join(args.output_directory, os.path.basename(infile))
        # always save as png
        outfile = "{}.png".format(os.path.splitext(outfile)[0])
        print("Saving: {}".format(outfile))
        do_faceswap(args.base_file, infile, outfile, args.tight_mask)

