import numpy as np
import cv2


def read_images(path_img, path_mask):
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    img = np.array(img, dtype='uint8')

    mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
    mask = mask * 255
    mask = np.array(mask, dtype='uint8')

    return img, mask


def crop_images(image, mask):
    """
    Function finds upper and lower bounds of plank mask
    and crops an image so that plank and background areas
    are approximately equal. This is done due to dataset 
    balancing purposes.
    """
    offset_per = 0.05
    stride = int(mask.shape[0] * 0.03)
    y_start = int(mask.shape[0] * offset_per)
    y_end = int(mask.shape[0] - y_start)
    n = (y_end-y_start) // stride
    lower_bound_flag = False
    upper_bound_flag = False

    for i in range(int(n)):
        if not lower_bound_flag:
            if np.max(mask[y_start, :]) == 1:
                lower_bound_flag = True
                lower_bound = y_start - stride
            else:
                y_start += stride

        if not upper_bound_flag:
            if np.max(mask[y_end, :]) == 1:
                upper_bound_flag = True
                upper_bound = y_end + stride
            else:
                y_end -= stride

    # Estimate height of plank area and offset based on that
    height = upper_bound - lower_bound
    off = height // 2
    y_l = lower_bound-off
    y_u = upper_bound+off
    # Check if new bounds don't extend over image boundaries
    if y_l < 0:
        y_l = 0
    if y_u > image.shape[0]:
        y_u = image.shape[0]

    # Crop image and mask according to determined bounds
    image_cropped = image[y_l:y_u, :]
    mask_cropped = mask[y_l:y_u, :]

    return image_cropped, mask_cropped


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
