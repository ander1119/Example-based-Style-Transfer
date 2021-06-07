"""
Featured-Based Image Metamorphosis
Beier 1992
"""
import numpy as np
from scipy import interpolate
import cv2
import matplotlib.pyplot as plt
from morpher import mls_affine_deformation

def draw(img):
    # img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)
    plt.plot()
    # print(abc.shape)
    plt.imshow(img)
    plt.show()

def imageMorpher(src, dst, pq1, pq2):
    warp, vx, vy  = mls_affine_deformation(src, dst, pq1, pq2)
    print(vx.shape, vx.max())
    print(vy.shape, vy.max())
    # draw(warp)
    return warp, vy, vx

def localMatching(style_img, input_img, style_mask, input_mask, vx, vy):
    tmpStyle, tmpInput = style_img.copy(), input_img.copy()
    tmpStyle[style_mask == 0], tmpInput[input_mask == 0] = 0, 0
    height, width, c = tmpInput.shape
    print(tmpInput.shape)
    n_layer = 7                               # 層數
    L_style, L_input = [], []                              # Laplacian stack of style and input              
    for i in range(n_layer):
        if i == 0:
            Gtmp_style = cv2.pyrDown(tmpStyle, None)
            Gtmp_input = cv2.pyrDown(tmpInput, None)
            L_style.append(tmpStyle - cv2.resize(Gtmp_style, (width, height)))
            L_input.append(tmpInput - cv2.resize(Gtmp_input, (width, height)))
        else:
            Gtmp_style = cv2.pyrDown(last_style, None)
            Gtmp_input = cv2.pyrDown(last_input, None)
            L_style.append(cv2.resize(last_style, (width, height)) - cv2.resize(Gtmp_style, (width, height)))
            L_input.append(cv2.resize(last_input, (width, height)) - cv2.resize(Gtmp_input, (width, height)))
        last_style, last_input = Gtmp_style, Gtmp_input
    resid_style = cv2.resize(last_style, (width, height))
    resid_input = cv2.resize(last_input, (width, height))
    S_style, S_input = [], []
    for i in range(n_layer):
        S_style_tmp = cv2.pyrDown(L_style[i] ** 2, None)
        S_input_tmp = cv2.pyrDown(L_input[i] ** 2, None)
        for j in range(i):
            S_style_tmp = cv2.pyrDown(S_style_tmp, None)
            S_input_tmp = cv2.pyrDown(S_input_tmp, None)
        S_style.append(cv2.resize(S_style_tmp, (width, height)))
        S_input.append(cv2.resize(S_input_tmp, (width, height)))
    eps = 0.01 ** 2
    gain_max = 2.8
    gain_min = 0.005
    output = np.zeros((height, width, c))
    for i in range(n_layer):
        gain = np.sqrt(np.divide(S_style[i], (S_input[i] + eps)))
        gain[gain <= gain_min] = 1
        gain[gain > gain_max] = gain_max
        output += np.multiply(L_input[i], gain)
    output += resid_style
    return output
    # draw(np.int32(resid_style))
    # draw(np.int32(resid_input))
    # for i in range(n_layer):
    #     draw(np.float32(L_style[i]))
    #     print(L_style[i].shape)
    #     print(L_input[i].shape)
    #     print("------------------")

def replaceBackground(matched, style, input, style_mask, input_mask, vx, vy):
    temp = np.zeros(input.shape, dtype=np.uint8)
    temp[style_mask == 0] = style[style_mask == 0]
    temp[input_mask == 255] = 0
    # xy = (255 - style_mask).astype(np.uint8)
    # bkg = cv.inpaint(temp, xy[:, :, 0], 10, cv.INPAINT_TELEA)
    # imsave('output/bkg.jpg', bkg.astype(int))
    # TODO: Extrapolate background
    xy = np.logical_not(input_mask.astype(bool))
    matched[xy] = 0
    output = temp + matched
    output[output > 255] = 255
    output[output <= 0] = 0
    output = output.astype(int)
    # print(output.shape)
    output[input_mask == 0] = 0
    cv2.imwrite('output/myTemp.jpg', output)
    output = cv2.cvtColor(np.float32(output), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 3))
    plt.subplot(131)
    plt.axis('off')
    plt.imshow(np.int32(cv2.cvtColor(np.float32(input), cv2.COLOR_BGR2RGB)))
    plt.title("Before")
    plt.subplot(132)
    plt.axis('off')
    plt.imshow(np.int32(style))
    plt.title("Style Image")
    plt.subplot(133)
    plt.axis('off')
    plt.imshow(np.int32(output))
    plt.title("After")
    plt.tight_layout(w_pad=0.1)
    plt.show()
    # imsave('output/temp.jpg', style.astype(int))
    return matched