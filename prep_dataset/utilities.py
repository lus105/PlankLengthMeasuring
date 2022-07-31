import h5py
import numpy as np
import os
import cv2


def read_images(path_img, path_mask):
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    img = np.array(img, dtype='uint8')
    (h1, h2, w1, w2) = crop(img)

    mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
    mask = mask*255.0

    cropped_mask = mask[h1:h2, w1:w2]
    cropped_img = img[h1:h2, w1:w2]
    return cropped_img, cropped_mask


def crop(image):
    # function finds ROI by summing pixel rows and columns
    image_img = image
    treshold = 100
    stride = 20
    offset_per = 0.05
    # along x axis
    x_start = int(image.shape[1]*offset_per)
    x_end = int(image.shape[1]-x_start)
    x_stride = (x_end-x_start)//stride

    for i in range(x_stride):
        max_pix = np.max(image_img[:, x_start])
        x_start += stride
        if max_pix > treshold:
            x_l = i
            break
    for i in range(x_stride):
        max_pix = np.max(image_img[:, x_end])
        x_end -= stride
        if max_pix > treshold:
            x_r = i
            break

    # along y axis
    y_start = int(image.shape[0]*offset_per)
    y_end = int(image.shape[0]-y_start)
    y_stride = (y_end-y_start)//stride

    for i in range(y_stride):
        max_pix = np.max(image_img[y_start, :])
        y_start += stride
        if max_pix > treshold:
            y_u = i
            break
    for i in range(y_stride):
        max_pix = np.max(image_img[y_end, :])
        y_end -= stride
        if max_pix > treshold:
            y_l = i
            break

    x_start = int(image.shape[1]*offset_per)
    x_end = int(image.shape[1]-x_start)
    y_start = int(image.shape[0]*offset_per)
    y_end = int(image.shape[0]-y_start)

    bounds = (y_start+stride*(y_u-10), y_end-stride*(y_l-10),
              x_start+stride*(x_l-10), x_end-stride*(x_r-10))
    #cropped_img = image_img[(y_start+stride*(y_u-10)):(y_end-stride*(y_l-10)),(x_start+stride*(x_l-10)):(x_end-stride*(x_r-10))]
    # print(cropped_img.shape)
    # cv2.imwrite('cropped.bmp',cropped_img)
    return bounds


def extract_patches(img, groundTruth, patch_h, patch_w):
    # normalization 0-1
    img = img/255.
    groundTruth = groundTruth/255.
    data_consistency_check(img, groundTruth)

    img_h = img.shape[0]  # height of the full image
    img_w = img.shape[1]  # width of the full image

    N_patches = int((img_h/patch_h)*(img_w/patch_w))
    N_patches_h = int(img_h/patch_h)
    N_patches_w = int(img_w/patch_w)

    patches_imgs_train = np.empty((N_patches, patch_h, patch_w))
    patches_gt_train = np.empty((N_patches, patch_h, patch_w))

    # iter over the total number of patches (N_patches)
    iter_tot = 0
    for y in range(N_patches_h):
        for x in range(N_patches_w):
            x_st = patch_w*(x)
            y_st = patch_h*(y)
            patch = img[y_st:y_st+int(patch_h), x_st:x_st+int(patch_w)]
            patch_gt = groundTruth[y_st:y_st +
                                   int(patch_h), x_st:x_st+int(patch_w)]

            patches_imgs_train[iter_tot] = patch
            patches_gt_train[iter_tot] = patch_gt
            iter_tot += 1
    data_consistency_check(img, groundTruth)

    return patches_imgs_train, patches_gt_train


def data_consistency_check(img, gtruth):
    assert(len(img.shape) == len(gtruth.shape))
    assert(img.shape[0] == gtruth.shape[0])
    assert(img.shape[1] == gtruth.shape[1])
