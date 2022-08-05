from __future__ import division
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from generator import DataGenerator
import models as M
import os
import sys
sys.path.append('..')
from PlankLengthMeasuring.settings import options
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
    model = M.basic_unet(
        input_size=(cfg.patch_size, cfg.patch_size, cfg.num_channels))
    model.summary()

    _, _, n_im_tr = next(os.walk(cfg.patch_imgs_dataset_path+'/train/'))
    img_ids_train = ['id-'+str(x+1) for x in range(len(n_im_tr))]
    _, _, n_ms_tr = next(os.walk(cfg.patch_masks_dataset_path+'/train/'))
    msk_ids_train = ['id-'+str(x+1) for x in range(len(n_ms_tr))]
    _, _, n_im_vl = next(os.walk(cfg.patch_imgs_dataset_path+'/val/'))
    img_ids_val = ['id-'+str(x+1) for x in range(len(n_im_vl))]
    _, _, n_ms_vl = next(os.walk(cfg.patch_masks_dataset_path+'/val/'))
    msk_ids_val = ['id-'+str(x+1) for x in range(len(n_ms_vl))]

    training_generator = DataGenerator(
        img_ids_train, msk_ids_train, val=False, **params)
    validation_generator = DataGenerator(
        img_ids_val, msk_ids_val, val=True, **params)

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