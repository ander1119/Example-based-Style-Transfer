import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2
import pickle
from face_alignment import FaceAlignment, LandmarksType
import sys
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

def mls_affine_deformation(styleImg, targetImg, p, q, alpha=0.8, density=1):
    height, width = styleImg.shape[0], styleImg.shape[1]
    allX, allY = np.linspace(0, width, num=width), np.linspace(0, height, num=height)
    vy, vx = np.meshgrid(allX, allY)
    vx, vy = vx.reshape(1, height, width), vy.reshape(1, height, width)

    ## 把 p, q 座標 x, y 交換（我們要將 styleImg p 轉到）
    p, q = q[:, [1, 0]], p[:, [1, 0]]
    
    ## 根據 p_i 給每一個點 v 權重 weight_i 
    v_reshaped = np.vstack((vx, vy))                                    # [2, height, width]
    p_reshaped = p.reshape((p.shape[0], 2, 1, 1))                      # [68, 2, 1, 1]
    q_reshaped = q.reshape((q.shape[0], 2, 1, 1))
    w = 1.0 / np.sum((p_reshaped - v_reshaped) ** 2, axis=1)**alpha     # [68, height, width]
    w[ w == np.inf ] = 2**32 - 1
    ## p_star, q_star, p_hat, q_hat
    w_sum = np.sum(w, axis=0)
    w_p = np.sum(w * p_reshaped.transpose(1, 0, 2, 3), axis=1)
    w_q = np.sum(w * q_reshaped.transpose(1, 0, 2, 3), axis=1)          
    p_star = w_p / w_sum                                                # [2, height, width]
    q_star = w_q / w_sum
    p_hat = p_reshaped - p_star                                         # [68, 2, height, width]
    q_hat = q_reshaped - q_star                                         # [68, 2, height, width]

    ## phat_T @ w @ phat for every i
    p_hat_T_reshaped = p_hat.reshape(p.shape[0], 2, 1, height, width)                            # [68, 2, 1, height, width]
    p_hat_reshaped = p_hat.reshape(p.shape[0], 1, 2, height, width)                              # [68, 1, 2, height, width]
    reshaped_w = w.reshape(p.shape[0], 1, 1, height, width)                                      # [68, 1, 1, height, width]
    phat_T_w_phat = p_hat_T_reshaped * reshaped_w * p_hat_reshaped                               # [68, 2, 2, height, width]
    phat_T_w_phat = np.sum(phat_T_w_phat, axis=0)                                                 # [2, 2, height, width]

    ## inverse of phat_T_w_phat
    # print(phat_T_w_phat.shape)
    # inv_phat_T_w_phat = np.linalg.inv(phat_T_w_phat.transpose(2, 3, 0, 1))                        # [grow, gcol, 2, 2]
    try:                
        inv_phat_T_w_phat = np.linalg.inv(phat_T_w_phat.transpose(2, 3, 0, 1))                            # [grow, gcol, 2, 2]
        flag = False                
    except np.linalg.linalg.LinAlgError:                
        flag = True             
        det = np.linalg.det(phat_T_w_phat.transpose(2, 3, 0, 1))                                 # [grow, gcol]
        det[det < 1e-8] = np.inf                
        reshaped_det = det.reshape(1, 1, height, width)                                    # [1, 1, grow, gcol]
        adjoint = phat_T_w_phat[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]               # [2, 2, grow, gcol]
        adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]                  # [2, 2, grow, gcol]
        inv_phat_T_w_phat = (adjoint / reshaped_det).transpose(2, 3, 0, 1)              # [grow, gcol, 2, 2]


    ## To be faster, Calculate "A" matrix
    mul_left = v_reshaped - p_star                                                       # [2, height, width]
    reshaped_mul_left = mul_left.reshape(1, 2, height, width).transpose(2, 3, 0, 1)      # [height, width, 1, 2]
    reshaped_mul_right = (reshaped_w * p_hat_T_reshaped).transpose(0, 3, 4, 1 ,2)        # [68, height, width, 2, 1]
    # print(reshaped_mul_left.shape, inv_phat_T_w_phat.shape, reshaped_mul_right.shape)
    A = reshaped_mul_left@inv_phat_T_w_phat@reshaped_mul_right                           # [68, height, width, 1, 1]
    reshaped_A = A.reshape(68, 1, height, width)                                         # [68, 1, height, width]
    # print(reshaped_A)
    # Get final image transfomer -- 3-D array
    transformers = np.sum(reshaped_A * q_hat, axis=0)+q_star                             # [2, height, width]
    # Correct the points where p_hat_T_w_p_hat is singular
    if flag:
        blidx = det == np.inf    # bool index
        transformers[0][blidx] = vx[blidx] + q_star[0][blidx] - p_star[0][blidx]
        transformers[1][blidx] = vy[blidx] + q_star[1][blidx] - p_star[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = np.ones_like(styleImg) * 255
    new_gridY, new_gridX = np.meshgrid((np.arange(width) / density).astype(np.int16), 
                                        (np.arange(height) / density).astype(np.int16))
    # transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]    # [grow, gcol]
    transformed_image[new_gridX, new_gridY] = styleImg[tuple(transformers.astype(np.int16))]
    # print(tuple(transformers.astype(np.int16).shape))
    # print("new_gridX", new_gridX)
    # print("new_gridY", new_gridY)
    # print("trans", transformers)
    return transformed_image, transformers[0].astype(np.int16), transformers[1].astype(np.int16)

if __name__ == "__main__":
    args = myArgsParser()
    style_dir = args.style_dir
    style_mask = style_dir + "/head_masks/" + args.style_img
    style_img = style_dir + "/" + args.style_img
    input_dir = args.input_dir
    if args.input_img:
        input_mask = input_dir + "/head_masks/" + args.input_img
        input_img = input_dir + "/" + args.input_img
        # print("style_dir", style_dir)
        # print("style_mask", style_mask)
        # print("style_img",style_img)
        # print("input_dir", input_dir)
        # print("input_mask", input_mask)
        # print("input_img", input_img)
        style_mask = np.float32(cv2.imread(style_mask))
        input_mask = np.float32(cv2.imread(input_mask))
        image, target = cv2.imread(style_img), cv2.imread(input_img)
        style_name = os.path.basename(style_img).split('.')[0]
        input_name = os.path.basename(input_img).split('.')[0]
        fa = FaceAlignment(LandmarksType._2D, device='cpu', flip_input=False)
        style_lm = fa.get_landmarks(image)[0]
        input_lm = fa.get_landmarks(target)[0]
        p = np.array(style_lm)
        q = np.array(input_lm)
        after_img, vx, vy = mls_affine_deformation(image, target, p, q, alpha=2, density=1)
        output_ori = cv2.imread("./code/Gpos/sample2.png")
        height, width = after_img.shape[0], after_img.shape[1]
        output_ori = cv2.resize(output_ori, (width,height))
        output_des = output_ori.copy()
        tmp_mask = style_mask.copy()
        new_gridY, new_gridX = np.meshgrid((np.arange(width)).astype(np.int16),  (np.arange(height)).astype(np.int16))
        output_des[new_gridX, new_gridY] = output_ori[vx, vy]
        tmp_mask[new_gridX, new_gridY] = style_mask[vx, vy]
        after_img[tmp_mask == 0] = 0
        image[style_mask == 0] = 0
        cv2.imwrite(style_dir  +"/G_pos/" + args.style_img, image)
        cv2.imwrite(input_dir  +"/G_pos/" + args.input_img, after_img)
        print("End of mlsAffineMorpher.")