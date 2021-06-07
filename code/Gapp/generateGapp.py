import os
import pickle
import cv2
import numpy as np
from face_alignment import FaceAlignment, LandmarksType
from myStyleTransferFunc import imageMorpher, localMatching, replaceBackground
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    style_dir = sys.argv[1]
    style_name = sys.argv[1]+"/"+sys.argv[2]
    style_mask = sys.argv[1]+"/head_masks/"+sys.argv[2]
    input_dir = sys.argv[3]
    input_name = sys.argv[3]+"/"+sys.argv[4]
    input_mask = sys.argv[3] + "/head_masks/"+sys.argv[4]
    # print(style_name, input_name)
    # print(style_mask, input_mask)
    # style_name, input_name = 'input/style5.jpg', 'input/color_ex.jpg'
    # style_mask, input_mask = 'input/mask_style5.jpg', 'input/mask_colorEx.jpg'
    style_img = np.float32(cv2.imread(style_name))
    input_img = np.float32(cv2.imread(input_name))
    style_mask = np.float32(cv2.imread(style_mask))
    input_mask = np.float32(cv2.imread(input_mask))
    style_name = os.path.basename(style_name).split('.')[0]
    input_name = os.path.basename(input_name).split('.')[0]
    fa = FaceAlignment(LandmarksType._2D, device='cpu', flip_input=False)
    style_lm = fa.get_landmarks(style_img)[0]
    input_lm = fa.get_landmarks(input_img)[0]
    # if os.path.exists('input/%s_%s_lm.pkl' % (style_name, input_name)):
    #     with open('input/%s_%s_lm.pkl' % (style_name, input_name), 'rb') as f:
    #         pkl = pickle.load(f)
    #         style_lm = pkl['style']
    #         input_lm = pkl['input']
    # else:
    #     fa = FaceAlignment(LandmarksType._2D, device='cpu', flip_input=False)
    #     style_lm = fa.get_landmarks(style_img)[0]
    #     input_lm = fa.get_landmarks(input_img)[0]
    #     with open('input/%s_%s_lm.pkl' % (style_name, input_name),
    #                 'wb') as f:
    #         pickle.dump({
    #             'style': style_lm,
    #             'input': input_lm
    #         }, f, protocol=2)
    
    warped, vx, vy = imageMorpher(style_img, input_img, style_lm, input_lm)
    # print("after ImageMorpher")
    # matched = localMatching(style_img,  input_img, style_mask, input_mask, vx, vy)
    height, width, c = input_img.shape
    new_gridY, new_gridX = np.meshgrid((np.arange(width)).astype(np.int16), 
                                        (np.arange(height)).astype(np.int16))
    style_mask[new_gridX, new_gridY] = style_mask[vy, vx]
    matched = localMatching(warped, input_img, style_mask, input_mask, vx, vy)
    matched = replaceBackground(matched, style_img, input_img, style_mask, input_mask, vx, vy)
    cv2.imwrite(style_dir+"/"+"G_app/"+sys.argv[2], style_img)
    cv2.imwrite(input_dir+"/"+"G_app/"+sys.argv[4], matched)
