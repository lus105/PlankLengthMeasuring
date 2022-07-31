import os
import numpy as np
from utilities import read_images, extract_patches
from options import Options


def main():
    cfg = Options().parse()
    idx = 1
    # List all files, directories in the path
    for path, subdir, file in os.walk(cfg.imgs_train_path):
        for i in range(len(file)):
            print(cfg.imgs_train_path+file[i])
            print(cfg.masks_train_path+file[i])
            # Check if image has corresponding mask (file names must be the same)
            if os.path.exists(cfg.masks_train_path+file[i]):
                img_train, mask_train = read_images(
                    cfg.imgs_train_path+file[i], cfg.masks_train_path+file[i])
            # # Patches of images and gtruths
            # patches_imgs_train, patches_gt_train = extract_patches(
            #     img_train, gt_train, cfg.patch_h, cfg.patch_w)
            # # Saving as numpy patches
            # for patch_img, patch_gt in zip(patches_imgs_train, patches_gt_train):
            #     np.save(os.path.join(cfg.patch_imgs_dataset_path,
            #             'id-'+str(idx)), patch_img)
            #     np.save(os.path.join(cfg.patch_masks_dataset_path,
            #             'id-'+str(idx)), patch_gt)
            #     idx += 1


if __name__ == '__main__':
    main()
