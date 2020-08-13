import numpy as np
import random
import os
import skimage.transform as ski_tf
import keras.backend as K
from keras.preprocessing.image import (
    ImageDataGenerator, DirectoryIterator,
    img_to_array, load_img, array_to_img
)


class MaskGenerator:
    def __init__(self, mask_size=(256, 256), box_size=(128, 128)):
        self.mask_size = mask_size
        self.box_size = box_size

    def _draw(self):
        raise NotImplementedError

    def flow(self, batch_size=32, total_size=-1, **draw_kwargs):
        residual = total_size
        below_zero = total_size <= 0
        while below_zero or residual > 0:
            n = batch_size if below_zero else min(batch_size, residual)
            samples = [self._draw(**draw_kwargs) for _ in range(n)]
            masks = np.asarray([mask for mask, box in samples])
            boxes = np.asarray([box for mask, box in samples])
            yield (masks, boxes)
            residual = total_size if below_zero else total_size - n


class FixMaskGenerator(MaskGenerator):

    def _draw(self, x=None, y=None, w=None, h=None):
        box = np.asarray([x, y, self.box_size[1], self.box_size[0]], dtype=np.int)
        p = x + ((self.box_size[1] - w) >> 1)
        q = y + ((self.box_size[0] - h) >> 1)
        mask = np.zeros(self.mask_size + (1,), dtype=np.float)
        mask[q:q+h, p:p+w, 0] = 1.0
        return (mask, box)


class RandomMaskGenerator(MaskGenerator):

    def __init__(self, max_size=(96, 96), min_size=(16, 16), **kwargs):
        super().__init__(**kwargs)
        self.max_size = max_size
        self.min_size = min_size

    def _draw(self, seed=None):
        y = np.random.randint(0, self.mask_size[0] - self.box_size[0] + 1)
        x = np.random.randint(0, self.mask_size[1] - self.box_size[1] + 1)
        box = np.asarray([x, y, self.box_size[1], self.box_size[0]], dtype=np.int)

        w = np.random.randint(self.min_size[1], self.max_size[1] + 1)
        h = np.random.randint(self.min_size[0], self.max_size[0] + 1)

        p = x + ((self.box_size[1] - w) >> 1)
        q = y + ((self.box_size[0] - h) >> 1)

        mask = np.zeros(self.mask_size + (1,), dtype=np.float)
        mask[q:q+h, p:p+w, 0] = 1.0

        return (mask, box)


class PatchedDirectoryIterator(DirectoryIterator):
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) +
                           self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=None)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(
                                                                      1e7),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


def preprocessing(img, target_size=256, margin=1.5):
    # `random.randint`: Return random integer in range[a, b], including both end points.
    itermid_size = random.randint(target_size, round(target_size * margin))
    rows, cols, _ = img.shape

    if rows <= cols:
        new_rows = itermid_size
        new_cols = round(float(itermid_size * cols) / float(rows))
    else:
        new_cols = itermid_size
        new_rows = round(float(itermid_size * rows) / float(cols))

    # The result will be normalized as well.
    resized_img = ski_tf.resize(img, (new_rows, new_cols), mode='reflect')

    x = random.randint(0, new_cols - target_size)
    y = random.randint(0, new_rows - target_size)

    return resized_img[y:y+target_size, x:x+target_size, :].copy()
