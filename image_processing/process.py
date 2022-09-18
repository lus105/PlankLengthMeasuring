import numpy as np
import cv2


def vscan_points(cnt_img, offset, stride):
    n = (cnt_img.shape[1] - 2 * offset) / stride
    v_cord = offset
    pnts_list = []
    for i in range(int(n)):
        if v_cord > cnt_img.shape[1]:
            break
        pnts = cnt_img[:, v_cord]
        pnts_fil = np.squeeze(np.where(pnts[:, 0] == 255), axis=0)
        pnts_fil = tuple(pnts_fil)
        if len(pnts_fil) != 0:
            pnts_list.append((v_cord, pnts_fil))
        v_cord += stride
    return pnts_list


def find_peaks(image, th_value, axis):
    image = image / 255.

    if axis == 0:
        iter_range = image.shape[1]
        sum = np.sum(image, axis=axis)
    else:
        iter_range = image.shape[0]
        sum = np.sum(image, axis=axis)

    lim1_b, lim2_b = False, False
    for i in range(0, iter_range, 100):
        if not lim1_b and (sum[i] > th_value):
            lim1 = i
            lim1_b = True
        if not lim2_b and (sum[-i] > th_value):
            lim2 = iter_range - i
            lim2_b = True
        if lim1_b and lim2_b:
            break
    return lim1, lim2


def crop(image, x_th, y_th, off):
    y_u, y_l = find_peaks(image, x_th, axis=1)
    x_l, x_r = find_peaks(image, y_th, axis=0)
    cropped_image = image[y_u-off:y_l+off, x_l-off:x_r+off]
    return cropped_image


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


def vertical_scan(img_shape, contours, offset, stride):
    empty = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.uint8)
    cv2.drawContours(empty, contours, -1, 255, 1)
    pnts = vscan_points(empty, offset, stride)
    empty = cv2.cvtColor(empty, cv2.COLOR_GRAY2BGR)
    for pnt in pnts:
        for pn in pnt[1]:
            cv2.circle(empty, [pnt[0], pn], 2, (0, 255, 0), -1)
    cv2.imwrite('points.png', empty)
    return empty
