import argparse
import os


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Paths
        self.parser.add_argument(
            '--imgs_train_path', type=str, default='dataset/images/')
        self.parser.add_argument(
            '--masks_train_path', type=str, default='dataset/masks/')
        self.parser.add_argument(
            '--patch_imgs_dataset_path', type=str, default='dataset/images_patch/')
        self.parser.add_argument(
            '--patch_masks_dataset_path', type=str, default='dataset/masks_patch/')
        self.parser.add_argument(
            '--train_split', type=str, default='train/')
        self.parser.add_argument(
            '--test_split', type=str, default='test/')
        self.parser.add_argument(
            '--val_split', type=str, default='val/')
        self.parser.add_argument(
            '--weight_path', type=str, default='models/weights/')
        self.parser.add_argument(
            '--results', type=str, default='dataset/results/')

        # Params
        self.parser.add_argument(
            '--patch_size', type=int, default=128)
        self.parser.add_argument(
            '--num_channels', type=int, default=1)

        # Training
        self.parser.add_argument(
            '--batch_size', type=int, default=32)
        self.parser.add_argument(
            '--shuffle', action='store_true', default=True)
        self.parser.add_argument(
            '--nb_epochs', type=int, default=100)
        self.parser.add_argument(
            '--weight_name', type=str, default='weights_no1.hdf5')
        self.parser.add_argument(
            '--split_ratio', type=tuple, default=(0.7, 0.2, 0.1))

        # Measuring consts
        self.parser.add_argument(
            '--x_th', type=int, default=500)
        self.parser.add_argument(
            '--y_th', type=int, default=500)
        self.parser.add_argument(
            '--crop_offset', type=int, default=200)
        self.parser.add_argument(
            '--scan_offset', type=int, default=200)
        self.parser.add_argument(
            '--mask_thres_val', type=int, default=150)
        self.parser.add_argument(
            '--cnt_min_per', type=int, default=7000)

    def parse(self):
        self.opt = self.parser.parse_args()
        self._create_directories()

        return self.opt

    def _create_directories(self):
        # Create directories for saving patches if they don't exist
        self.opt.patch_imgs_dataset_path = self.opt.patch_imgs_dataset_path + \
            str(self.opt.patch_size) + '/'
        self.opt.patch_masks_dataset_path = self.opt.patch_masks_dataset_path + \
            str(self.opt.patch_size) + '/'
        if not os.path.exists(self.opt.patch_imgs_dataset_path):
            os.makedirs(self.opt.patch_imgs_dataset_path)
        if not os.path.exists(self.opt.patch_masks_dataset_path):
            os.makedirs(self.opt.patch_masks_dataset_path)
