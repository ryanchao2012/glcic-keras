import os
import warnings
import random
import numpy as np
import skimage.io as ski_io
from keras.layers import Input, Dense
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from .models import DiscriminatorBuilder, _DisBuilder
from .helpers import RandomMaskGenerator, FixMaskGenerator
from .envs import (
    PJ, data_dir, evaluate_dir, ckpt_dir,
    generator_loss, discriminator_loss,
)

os.makedirs(evaluate_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)


def training(x_train, x_test=None, init_iters=1,
             eval_iters=50, ckpt_iters=100, max_iters=-1,
             pretrained_file=PJ(ckpt_dir, 'test_discriminator.h5')):

    input_tensor = Input(shape=(128, 128, 3), name='raw_image')
    color_prior = np.asarray([128, 128, 128], dtype=np.float) / 255.0

    discriminator_builder = _DisBuilder(activation='relu', metrics=['acc'],
                                        loss=discriminator_loss, debug=True,
                                        pretrained_file=pretrained_file)

    dis = discriminator_builder(input_tensor)
    h = dis(input_tensor)
    output_tensor = Dense(1, activation='sigmoid')(h)

    discriminator_net = discriminator_builder.compile(Model(input_tensor, output_tensor))

    batch_size = x_train.shape[0]
    datagan = ImageDataGenerator(rotation_range=20, horizontal_flip=True,
                                 fill_mode='reflect',
                                 width_shift_range=0.2, height_shift_range=0.2)
    maskgan = RandomMaskGenerator(mask_size=(256, 256), box_size=(128, 128),
                                  max_size=(96, 96), min_size=(16, 16))

    # Suppress trainable weights and collected trainable inconsistency warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        for i, (images, (masks, bboxes)) in enumerate(zip(datagan.flow(x_train, batch_size=batch_size),
                                                          maskgan.flow(batch_size=batch_size)), init_iters):
            fake_images = images * (1.0 - masks) + masks * color_prior
            cropped_fakes = [image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                             for image, bbox in zip(fake_images, bboxes)]
            cropped_reals = [image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                             for image, bbox in zip(images, bboxes)]
            d = cropped_fakes + cropped_reals
            cropped = np.asarray(d)
            labels = np.asarray([[lb] for lb in ([0] * fake_images.shape[0] + [1] * images.shape[0])])

            # ['loss', 'acc']
            d_loss, acc = discriminator_net.train_on_batch(cropped, labels)

            print(f'Iter: {i:05},\t Loss: {d_loss:.3E}, Accuracy: {acc:2f}', flush=True)
            if i % ckpt_iters == 0:
                discriminator_net.save(PJ(ckpt_dir, 'tes_discriminator.h5'))

            if max_iters > 0 and i >= max_iters:
                break


if __name__ == '__main__':
    eval_mask = list(RandomMaskGenerator().flow(total_size=1))[0][0]
    input_image = ski_io.imread(PJ(data_dir, 'food.jpg')).astype(np.float) / 255.0
    x_train = np.asarray([input_image])
    training(x_train, x_test=x_train, max_iters=1)
