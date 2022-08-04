from __future__ import division
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from generator import DataGenerator
import models as M
import os
import sys
sys.path.append('..')
from PlankLengthMeasuring.options import Options
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # Load settings from Options class
    cfg = Options().parse()
    params = {
        'patch_imgs_dataset_path': cfg.patch_imgs_dataset_path + 'train/',
        'patch_masks_dataset_path': cfg.patch_masks_dataset_path + 'train/',
        'dim': (cfg.patch_size, cfg.patch_size),
        'batch_size': cfg.batch_size,
        'n_channels': cfg.num_channels,
        'shuffle': cfg.shuffle}

    # Model definition
    model = M.basic_unet(
        input_size=(cfg.patch_size, cfg.patch_size, cfg.num_channels))
    model.summary()

    img_ids = ['id-'+str(x+1) for x in range(4849)]
    msk_ids = ['id-'+str(x+1) for x in range(4849)]

    training_generator = DataGenerator(
        img_ids[0:3879], msk_ids[0:3879], **params)
    validation_generator = DataGenerator(
        img_ids[3879:4849], msk_ids[3849:4849], **params)

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