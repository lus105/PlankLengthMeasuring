import numpy as np
import cv2


def read_images(path_img, path_mask):
    """
    Function reads image and mask followed by
    numpy array conversion
    """
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    img = np.array(img, dtype='uint8')

    mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
    mask = np.array(mask, dtype='uint8')
    #mask = mask * 255.

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
            if np.max(mask[y_start, :]) == 255:
                lower_bound_flag = True
                lower_bound = y_start - stride
            else:
                y_start += stride

        if not upper_bound_flag:
            if np.max(mask[y_end, :]) == 255:
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


def extract_patches(img, mask, patch_size):
    """
    Function chops given images into array of patches.
    """
    # Normalization 0-1
    #img = img / 255.
    #mask = mask / 255.
    data_consistency_check(img, mask)
    # Image size
    img_h, img_w = img.shape[:2]
    # Overlap size in pixels
    overlap = patch_size // 2
    N_patches_h = (2 * img_h - overlap) // patch_size
    N_patches_w = (2 * img_w - overlap) // patch_size
    patches_imgs_train = []
    patches_mask_train = []

    for y in range(N_patches_h):
        y_st = overlap * y
        if (y_st + patch_size) > img_h:
            y_st = y_st - patch_size
        for x in range(N_patches_w):
            x_st = overlap * x
            if (x_st + patch_size) > img_w:
                x_st = x_st - patch_size
            patch = img[y_st:y_st + patch_size, x_st:x_st + patch_size]
            patch_msk = mask[y_st:y_st + patch_size, x_st:x_st + patch_size]
            patches_imgs_train.append(patch)
            patches_mask_train.append(patch_msk)

    patches_imgs_train = np.array(patches_imgs_train)
    patches_mask_train = np.array(patches_mask_train)

    return patches_imgs_train, patches_mask_train


def data_consistency_check(img, mask):
    assert(len(img.shape) == len(mask.shape))
    assert(img.shape[0] == mask.shape[0])
    assert(img.shape[1] == mask.shape[1])
