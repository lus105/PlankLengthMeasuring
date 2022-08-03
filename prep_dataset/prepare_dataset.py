import os
import numpy as np
import cv2
from options import Options
from utilities import read_images, crop_images, extract_patches


def main():
    cfg = Options().parse()
    idx = 1
    # List all files, directories in the path
    for path, subdir, file in os.walk(cfg.imgs_train_path):
        for i in range(len(file)):
            print(cfg.imgs_train_path+file[i])
            print(cfg.masks_train_path+file[i])
            # Check if image has corresponding mask
            if os.path.exists(cfg.masks_train_path+file[i]):
                img_train, mask_train = read_images(
                    cfg.imgs_train_path+file[i], cfg.masks_train_path+file[i])
                # Cropping images so that each of them contains
                # aproximately equal areas of planks/background
                img_train_c, mask_train_c = crop_images(img_train, mask_train)
                # Ectracting patches
                patches_imgs_train, patches_mask_train = extract_patches(
                    img_train_c, mask_train_c, cfg.patch_size)
                # Saving as numpy patches
                for patch_img, patch_mask in zip(patches_imgs_train, patches_mask_train):
                    cv2.imwrite(os.path.join(cfg.patch_imgs_dataset_path,
                                             'id-'+str(idx))+'.png', patch_img)
                    cv2.imwrite(os.path.join(cfg.patch_masks_dataset_path,
                                             'id-'+str(idx))+'.png', patch_mask)
                    idx += 1


if __name__ == '__main__':
    main()
