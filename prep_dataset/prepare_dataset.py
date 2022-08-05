import os
import numpy as np
import cv2
import sys
sys.path.append('..')
from PlankLengthMeasuring.settings import options
from utilities import read_images, crop_images, extract_patches


def main():
    cfg = options.Options().parse()
    _, _, file_m = next(os.walk(cfg.masks_train_path))
    split_ratio = cfg.split_ratio
    num_file_m = len(file_m)
    split_num = np.arange(3)
    split_tp = ['train', 'test', 'val']
    for i in range(len(split_num)):
        split_num[i] = int(num_file_m * split_ratio[i])
        if not os.path.exists(cfg.patch_imgs_dataset_path+'/'+split_tp[i]):
            os.makedirs(cfg.patch_imgs_dataset_path+'/'+split_tp[i])
        if not os.path.exists(cfg.patch_masks_dataset_path+'/'+split_tp[i]):
            os.makedirs(cfg.patch_masks_dataset_path+'/'+split_tp[i])
    # Adding image to test set if splitting results in floating point division
    if np.sum(split_num) != num_file_m:
        split_num[1] += 1
    # List all files, directories in the path
    idx = 1
    curr_split = 0
    counter = 0
    for _, _, file in os.walk(cfg.imgs_train_path):
        for i in range(len(file)):
            print(cfg.imgs_train_path+file[i])
            print(cfg.masks_train_path+file[i])
            # Check if image has corresponding mask
            if os.path.exists(cfg.masks_train_path+file[i]):
                counter +=1
                if counter > split_num[curr_split]:
                    counter = 0
                    curr_split += 1
                    idx = 1
                img_train, mask_train = read_images(
                    cfg.imgs_train_path+file[i], cfg.masks_train_path+file[i])
                # Cropping images so that each of them contains
                # aproximately equal areas of planks/background
                img_train_c, mask_train_c = crop_images(img_train, mask_train)
                # Ectracting patches
                patches_imgs_train, patches_mask_train = extract_patches(
                    img_train_c, mask_train_c, cfg.patch_size)
                for patch_img, patch_mask in zip(patches_imgs_train, patches_mask_train):
                    cv2.imwrite((cfg.patch_imgs_dataset_path+'/'+split_tp[curr_split]+'/id-'+str(idx)+'.png'), patch_img)
                    cv2.imwrite((cfg.patch_masks_dataset_path+'/'+split_tp[curr_split]+'/id-'+str(idx)+'.png'), patch_mask)
                    idx += 1


if __name__ == '__main__':
    main()
