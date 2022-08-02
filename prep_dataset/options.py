import argparse
import os


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # PATHS
        self.parser.add_argument(
            '--imgs_train_path', type=str, default='dataset/images/')
        self.parser.add_argument(
            '--masks_train_path', type=str, default='dataset/masks/')
        self.parser.add_argument(
            '--patch_imgs_dataset_path', type=str, default='dataset/images_patch/')
        self.parser.add_argument(
            '--patch_masks_dataset_path', type=str, default='dataset/masks_patch/')

        # PARAMS
        self.parser.add_argument(
            '--patch_size', type=int, default=128)
        self.parser.add_argument(
            '--num_channels', type=int, default=1)

    def parse(self):
        self.opt = self.parser.parse_args()
        self._create_directories()

        return self.opt

    def _create_directories(self):
        # Create directories for saving patches if they don't exist
        if not os.path.exists(self.opt.patch_imgs_dataset_path+str(self.opt.patch_size)):
            os.makedirs(self.opt.patch_imgs_dataset_path+str(self.opt.patch_size))
        if not os.path.exists(self.opt.patch_masks_dataset_path+str(self.opt.patch_size)):
            os.makedirs(self.opt.patch_masks_dataset_path+str(self.opt.patch_size))