"""
2022 Summer
@author: Lukas Zabulis
Github: https://github.com/lus105
"""
import cv2
import os
import tensorflow as tf

from models.model import U_Net
from settings import options
from prep_dataset import utilities
from image_processing import process

# GPU CONFIG
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#tf.device('/cpu:0')


def main():
    # Load config file
    cfg = options.Options().parse()
    # Load model
    model = U_Net((cfg.patch_size, cfg.patch_size, cfg.num_channels))
    model.load_weights(os.path.join(cfg.weight_path, cfg.weight_name))
    # Iterate trough testing images
    image_paths = utilities.gather_image_from_dir(cfg.imgs_train_path)
    for name in image_paths:
        image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        file_name = utilities.get_file_name(name)
        # Crop image along both axis
        cropped_img = process.crop(image, cfg.x_th, cfg.y_th, cfg.crop_offset)
        # Extract patches
        img_stack, n_patches_h, n_patches_w = utilities.extract_patches_test(cropped_img, cfg.patch_size)
        # Segment patches
        seg_stack = model.predict(img_stack, batch_size=10, verbose=1)
        # Recompone image from segmented patches
        recomponed_image = utilities.recompone(seg_stack, n_patches_h, n_patches_w, image.shape, cfg.patch_size)
        '''Measuring (image processing)'''
        mask_th = process.mask_preprocessing(recomponed_image, cfg.mask_thres_val)
        contours = process.find_contours(mask_th, cfg.cnt_min_per)
        contours_img = process.vertical_scan(cropped_img.shape, contours)
        cv2.imwrite(cfg.results + file_name + '_cr_seg.png', contours_img)


if __name__ == '__main__':
    main()
