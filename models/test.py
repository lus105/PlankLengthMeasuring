# Pixel acc
# IoU
# Dice (F1 Score)

from __future__ import division
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from generator import DataGenerator
from model import U_Net
import os
import sys
sys.path.append('..')
from PlankLengthMeasuring.settings import options
from PlankLengthMeasuring.prep_dataset import utilities
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # Load settings from Options class
    cfg = options.Options().parse()
    params = {
        'patch_imgs_dataset_path': cfg.patch_imgs_dataset_path,
        'patch_masks_dataset_path': cfg.patch_masks_dataset_path,
        'dim': (cfg.patch_size, cfg.patch_size),
        'batch_size': cfg.batch_size,
        'n_channels': cfg.num_channels,
        'shuffle': cfg.shuffle}

    # Model definition
    model = U_Net(input_size=(cfg.patch_size, cfg.patch_size, cfg.num_channels))

    test_img_id = utilities.get_file_list(cfg.patch_imgs_dataset_path, cfg.test_split)
    test_msk_id = utilities.get_file_list(cfg.patch_masks_dataset_path, cfg.test_split)

    test_generator_x = DataGenerator(test_img_id, test_msk_id, split=cfg.test_split, **params)
    history = model.evaluate(x=test_generator_x, verbose=1, use_multiprocessing=True)


if __name__ == '__main__':
    main()
