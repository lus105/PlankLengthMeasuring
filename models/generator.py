import numpy as np
import keras
import cv2


class DataGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, val, patch_imgs_dataset_path, patch_masks_dataset_path,
                 batch_size=8, dim=(128, 128), n_channels=1, shuffle=True):
        'Initialization'
        self.patch_imgs_dataset_path = patch_imgs_dataset_path
        self.patch_masks_dataset_path = patch_masks_dataset_path
        self.val = val
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size, self.n_channels, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if not self.val:
                split = '/train/'
            else:
                split = '/val/'

            X[i, ] = cv2.imread(self.patch_imgs_dataset_path + split + ID + '.png', cv2.IMREAD_GRAYSCALE) / 255.
            y[i, ] = cv2.imread(self.patch_masks_dataset_path + split + ID + '.png', cv2.IMREAD_GRAYSCALE) / 255.

        X = np.einsum('klij->kijl', X)
        y = np.einsum('klij->kijl', y)
        return X, y
