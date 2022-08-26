"""
2022 Summer
@author: Lukas Zabulis
Github: https://github.com/lus105
"""

import tensorflow as tf
from keras.models import load_model
import cv2
import os
from models.model import U_Net
from settings import options
from prep_dataset import utilities

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

    image_paths = utilities.gather_image_from_dir(cfg.imgs_train_path)

    for name in image_paths:
        image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        file_name = utilities.get_file_name(name)
        '''Crop image'''
        img_stack, n_patches_h, n_patches_w = utilities.extract_patches_test(image, cfg.patch_size)
        predictions = model.predict(img_stack, batch_size=10, verbose=1)
        recomponed_image = utilities.recompone(predictions, n_patches_h, n_patches_w, image.shape, cfg.patch_size)
        '''Measuring (image processing)'''
        cv2.imwrite(cfg.results + file_name + '.png', recomponed_image)


if __name__ == '__main__':
    main()
