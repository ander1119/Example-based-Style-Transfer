import os
import pickle
import cv2
import numpy as np
from face_alignment import FaceAlignment, LandmarksType
from myStyleTransferFunc import imageMorpher, localMatching, replaceBackground
import matplotlib.pyplot as plt
import sys
import argparse
def myArgsParser():
    parser = argparse.ArgumentParser(description="mlsAffineMorpher")
    parser.add_argument(
        "-id",
        "--input_dir",
        type=str,
        help="Directory of input image",
        required=True
    )
    parser.add_argument(
        "-im",
        "--input_img",
        type=str,
        help="One input image"
    )
    parser.add_argument(
        "-sd",
        "--style_dir",
        type=str,
        help="Directory of style image",
        required=True
    )
    parser.add_argument(
        "-sm",
        "--style_img",
        type=str,
        help="One style image"
    )
    args = parser.parse_args()
    # print(args)
    return args
if __name__ == '__main__':
    args = myArgsParser()
    style_dir = args.style_dir
    style_mask = style_dir + "/head_masks/" + args.style_img
    style_img = style_dir + "/" + args.style_img
    input_dir = args.input_dir
    if args.input_img:
        input_mask = input_dir + "/head_masks/" + args.input_img
        input_img = input_dir + "/" + args.input_img
        style_img = np.float32(cv2.imread(style_img))
        input_img = np.float32(cv2.imread(input_img))
        style_mask = np.float32(cv2.imread(style_mask))
        input_mask = np.float32(cv2.imread(input_mask))
        fa = FaceAlignment(LandmarksType._2D, device='cpu', flip_input=False)
        style_lm = fa.get_landmarks(style_img)[0]
        input_lm = fa.get_landmarks(input_img)[0]  
        warped, vx, vy = imageMorpher(style_img, input_img, style_lm, input_lm)
        # print("after ImageMorpher")
        # matched = localMatching(style_img,  input_img, style_mask, input_mask, vx, vy)
        height, width, c = input_img.shape
        new_gridY, new_gridX = np.meshgrid((np.arange(width)).astype(np.int16), 
                                            (np.arange(height)).astype(np.int16))
        # style_mask_copy = style_mask.copy()                                   
        # style_mask_copy[new_gridX, new_gridY] = style_mask[vy, vx]
        # matched = localMatching(warped, input_img, style_mask_copy, input_mask, vx, vy)
        matched = localMatching(style_img, input_img, style_mask, input_mask, vx, vy)
        matched = replaceBackground(matched, style_img, input_img, style_mask, input_mask, vx, vy)
        style_img[style_mask == 0] = 0
        cv2.imwrite(style_dir+"/G_app/"+args.style_img, style_img)
        cv2.imwrite(input_dir+"/G_app/"+args.input_img, matched)
        print("End of generateGapp.")