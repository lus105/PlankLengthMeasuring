from __future__ import division
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from generator import DataGenerator
from model import U_Net
import os
import glob
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

    train_img_id = utilities.get_file_list(cfg.patch_imgs_dataset_path, cfg.train_split)
    train_msk_id = utilities.get_file_list(cfg.patch_masks_dataset_path, cfg.train_split)
    val_img_id = utilities.get_file_list(cfg.patch_imgs_dataset_path, cfg.val_split)
    val_msk_id = utilities.get_file_list(cfg.patch_masks_dataset_path, cfg.val_split)

    training_generator = DataGenerator(
        train_img_id, train_msk_id, split=cfg.train_split, **params)
    validation_generator = DataGenerator(
        val_img_id, val_msk_id, split=cfg.val_split, **params)

    mcp_save = ModelCheckpoint(
        os.path.join(cfg.weight_path, cfg.weight_name), save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    history = model.fit(x=training_generator,
                        validation_data=validation_generator,
                        epochs=cfg.nb_epochs,
                        verbose=1,
                        callbacks=[mcp_save, reduce_lr_loss])

if __name__ == '__main__':
    main()