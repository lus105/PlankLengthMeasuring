import numpy as np
import cv2


def mask_preprocessing(image_mask, threshold_value):
    _, mask_thres = cv2.threshold(image_mask, threshold_value, 255, 0)
    holes = cv2.bitwise_not(mask_thres)
    cv2.floodFill(holes, None, (0, 0), 0)
    mask_thres = cv2.bitwise_or(mask_thres, holes)
    kernel_er = np.ones((3, 3), np.uint8)
    mask_thres = cv2.erode(mask_thres, kernel_er, iterations=1)
    return mask_thres


def find_contours(img, min_perimeter):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_new = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > min_perimeter:
            contours_new.append(cnt)
    return contours_new


def vertical_scan(img_shape, contours):
    empty = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.uint8)
    cv2.drawContours(empty, contours, -1, 255, 1)
    return empty
