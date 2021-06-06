import os
import argparse
import torch
import numpy as np
import face_alignment
import urllib.request
import ssl
import cv2
import matplotlib.pyplot as plt

from torchvision import transforms

IMG_EXT = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')

CLASS_MAP = {"background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5, "bus": 6, "car": 7,
             "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14, "person": 15,
             "potted plant": 16, "sheep": 17, "sofa": 18, "train": 19, "tv/monitor": 20}


def trimap(probs, size, conf_threshold):
    """
    This function creates a trimap based on simple dilation algorithm
    Inputs [3]: an image with probabilities of each pixel being the foreground, size of dilation kernel,
    foreground confidence threshold
    Output    : a trimap
    """
    mask = (probs > 0.05).astype(np.uint8) * 255

    pixels = 2 * size + 1
    kernel = np.ones((pixels, pixels), np.uint8)

    dilation = cv2.dilate(mask, kernel, iterations=1)

    remake = np.zeros_like(mask)
    remake[dilation == 255] = 127  # Set every pixel within dilated region as probably foreground.
    remake[probs > conf_threshold] = 255  # Set every pixel with large enough probability as definitely foreground.

    return remake


def parse_args():
    parser = argparse.ArgumentParser(description="Deeplab Segmentation")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Directory to save the output results. (required)",
    )
    parser.add_argument(
        "--target_class",
        type=str,
        default='person',
        choices=CLASS_MAP.keys(),
        help="Type of the foreground object.",
    )
    parser.add_argument(
        "--show",
        action='store_true',
        help="Use to show results.",
    )
    parser.add_argument(
        "--reprocess",
        action='store_true',
        help="Use to reprocess all images",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default='0.95',
        help="Confidence threshold for the foreground object. "
             "You can play with it to get better looking trimaps.",
    )

    args = parser.parse_args()
    return args


def main(input_dir, target_class, show, conf_threshold, reprocess):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()

    trimaps_path = os.path.join(input_dir, "trimaps")
    masks_path = os.path.join(input_dir, "masks")
    # head_trimaps_path = os.path.join(input_dir, "head_trimaps_path")
    os.makedirs(trimaps_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)
    # os.makedirs(head_trimaps_path, exist_ok=True)

    images_list = os.listdir(input_dir)
    for filename in images_list:
        trimap_filename = os.path.basename(filename).split('.')[0] + '.png'
        if not reprocess and os.path.isfile(os.path.join(trimaps_path, trimap_filename)):
            continue
        if not filename.endswith(IMG_EXT):
            continue
        print("extracting forefround and background of {}".format(filename))
        input_image = cv2.imread(os.path.join(input_dir, filename))
        original_image = input_image.copy()

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        with torch.no_grad():
            output = model(input_batch)['out'][0]
            output = torch.softmax(output, 0)

        output_cat = output[CLASS_MAP[target_class], ...].numpy()

        trimap_image = trimap(output_cat, 7, conf_threshold)

        output_cat = cv2.normalize(output_cat, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(trimaps_path, trimap_filename), trimap_image)
        cv2.imwrite(os.path.join(masks_path, trimap_filename), output_cat)

        if show:
            _, ax = plt.subplots(1,3)
            ax[0].imshow(output_cat)
            ax[1].imshow(original_image)
            ax[2].imshow(trimap_image)
            plt.show()
            # cv2.imshow('mask', output_cat)
            # cv2.imshow('image', original_image)
            # cv2.imshow('trimap', trimap_image)
    
    filenames = os.listdir(masks_path)
    kernel = np.ones((10, 10), np.int8)
    for filename in filenames:
        mask = cv2.imread(os.path.join(masks_path, filename), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(os.path.join(input_dir, filename))

        mask[mask < 240] = 0
        # plt.imshow(cv2.bitwise_and(image, image, mask=mask))
        # plt.show()
        # mask[mask != 255] = 0

        # image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # mask_hsv = cv2.inRange(image_hsv, (0, 40, 0), (25,255,255))
        # mask_YCrCb = cv2.inRange(image_YCrCb, (0, 138, 67), (255,173,133)) 
        # binary_mask = cv2.add(mask_hsv, mask_YCrCb)
        # image_foreground = cv2.erode(binary_mask, None, iterations=3)
        # dilated_binary_image = cv2.dilate(binary_mask, None, iterations=3)
        # ret, image_background = cv2.threshold(dilated_binary_image, 1, 128, cv2.THRESH_BINARY)
        # image_marker = cv2.add(image_foreground, image_background) 
        # image_marker = np.int32(image_marker)
        # cv2.watershed(image, image_marker)
        # m = cv2.convertScaleAbs(image_marker)
        # ret, image_mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # image_mask = cv2.bitwise_and(image_mask, mask)

        # mask_result = cv2.bitwise_and(image, image, mask = image_mask)
        # # mask_result = cv2.bitwise_and(mask_result, mask_result, mask = mask)

        # _, ax = plt.subplots(3, 2)
        # ax[0, 0].set_title("origin image")
        # ax[0, 0].imshow(image)
        # ax[0, 1].set_title("masked image")
        # ax[0, 1].imshow(mask_result)
        # ax[1, 0].set_title("hsv mask")
        # ax[1, 0].imshow(mask_hsv)
        # ax[1, 1].set_title("YCrCb mask")
        # ax[1, 1].imshow(mask_YCrCb)
        # ax[2, 0].set_title("hsv image")
        # ax[2, 0].imshow(image_hsv)
        # ax[2, 1].set_title("YCrCb image")
        # ax[2, 1].imshow(image_YCrCb)
        # # plt.imshow(image_YCrCb)
        # plt.show()

        print("aligning face of {}".format(filename))

        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu') 
        preds = fa.get_landmarks(image)
        preds = preds[0]
        head_mask = np.zeros_like(mask)
        head_center = (int(preds[:, 0].mean()), int(preds[:, 1].mean()))
        
        landmark_image = image.copy()
        for i in range(preds.shape[0]):
            cv2.circle(landmark_image, (int(preds[i][0]), int(preds[i][1])), 5, (255, 0, 0), -1)
        plt.imshow(landmark_image)
        plt.show()

        col_arr = np.arange(image.shape[0])
        col_arr = np.transpose(np.tile(col_arr, (image.shape[1], 1)))
        row_arr = np.arange(image.shape[1]).reshape(1, -1)
        row_arr = np.tile(row_arr, (image.shape[0], 1))

        print("cutting neck of {}".format(filename))

        for i in range(4, 12):
            # cv2.circle(image, (int(preds[i][0]), int(preds[i][1])), 5, (255, 0, 0), -1)
            # cv2.line(image_mask, (preds[i][0], preds[i][1]), (preds[i+1][0], preds[i+1][1]), 0, 4)
            new_mask = np.zeros_like(head_mask)
            m = (preds[i][1] - preds[i+1][1]) / (preds[i][0] - preds[i+1][0])
            b = -1 * m * preds[i][0] + preds[i][1]
            side_determination = (row_arr * m - col_arr - m * preds[i][0] + preds[i][1])
            side_determination *= side_determination[head_center]
            new_mask[side_determination < 0] = 255
            new_mask[side_determination >= 0] = 0
            head_mask = cv2.bitwise_or(head_mask, new_mask)

        head_mask = cv2.bitwise_and(mask, cv2.bitwise_not(head_mask))
        head_image = cv2.bitwise_and(image, image, mask=head_mask)

        # plt.imshow(head_mask)
        # plt.show()
        show_head = head_image.copy()
        show_head[show_head.sum(axis=2) == 0] = 255
        plt.imshow(show_head)
        plt.show()

        print("segmenting skin of {}".format(filename))

        image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(image_hsv, (0, 40, 0), (25,255,255))
        mask_YCrCb = cv2.inRange(image_YCrCb, (0, 138, 67), (255,173,133)) 
        skin_mask = cv2.bitwise_and(mask_hsv, mask_YCrCb, mask = head_mask)
        skin_image = cv2.bitwise_and(image, image, mask = skin_mask)

        show_skin = skin_image.copy()
        show_skin[show_skin.sum(axis=2) == 0] = 255
        plt.imshow(show_skin)
        plt.show()

        print("segmenting other features of {}".format(filename))

        left_eye_mask = np.zeros_like(mask)
        left_eye = preds[36:42]
        # left_eye_center = (int(left_eye[:, 0].mean()), int(left_eye[:, 1].mean()))
        left_eye_ellipse = cv2.fitEllipse(left_eye)
        # cv2.ellipse(image, left_eye_ellipse, (255, 255, 255), -1)
        
        right_eye_mask = np.zeros_like(mask)
        right_eye = preds[42:48]
        # right_eye_center = (int(right_eye[:, 0].mean()), int(right_eye[:, 1].mean()))
        right_eye_ellipse = cv2.fitEllipse(right_eye)

        lip = preds[48:60]
        lip_ellipse = cv2.fitEllipse(lip)

        mouth = preds[60:68]
        mouth_ellipse = cv2.fitEllipse(mouth)

        color_dict = {'skin': (124,252,0),
                      'hair': (255,140,0),
                      'eyes': (0,0,255),
                      'eyebrow': (255,0,0),
                      'nose': (255,255,0),
                      'lip': (135,206,250),
                      'mouth': (255,0,255)}

        result_image = np.zeros_like(image)
        result_image[head_mask != 0] = color_dict['hair']
        result_image[skin_mask != 0] = color_dict['skin']
        cv2.ellipse(result_image, left_eye_ellipse, color_dict['eyes'], -1)
        cv2.ellipse(result_image, left_eye_ellipse, color_dict['eyes'], 4)
        cv2.ellipse(result_image, right_eye_ellipse, color_dict['eyes'], -1)
        cv2.ellipse(result_image, right_eye_ellipse, color_dict['eyes'], 4)
        # cv2.ellipse(result_image, lip_ellipse, color_dict['lip'], -1)
        # cv2.ellipse(result_image, mouth_ellipse, color_dict['mouth'], -1)
        cv2.fillConvexPoly(result_image, lip.astype(int), color_dict['lip'])
        cv2.fillConvexPoly(result_image, mouth.astype(int), color_dict['mouth'])
        cv2.fillConvexPoly(result_image, preds[30:36].astype(int), color_dict['nose'])
        cv2.fillConvexPoly(result_image, preds[17:22].astype(int), color_dict['eyebrow'])
        cv2.fillConvexPoly(result_image, preds[22:27].astype(int), color_dict['eyebrow'])
        cv2.line(result_image, tuple(preds[27]), tuple(preds[30]), color_dict['nose'], 4)
        # cv2.fillPoly(result_image, [lip], color_dict['lip'])
        
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(head_image)
        ax[1].imshow(result_image)
        plt.show()



if __name__ == "__main__":
    args = parse_args()
    main(args.input_dir, args.target_class, args.show, args.conf_threshold, args.reprocess)