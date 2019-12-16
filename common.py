from enum import Enum

import tensorflow as tf
from tensorflow.keras import models, layers

BUFFER_SIZE = 10000


class ModelArchitecture(Enum):
    SA_MIRI = 1
    RESNET101 = 2
    MOBILENET = 3

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return ModelArchitecture[s.upper()]
        except KeyError:
            return s


def get_model(show_summary=True, architecture=ModelArchitecture.SA_MIRI):
    model = None

    if architecture == ModelArchitecture.SA_MIRI:
        model = models.Sequential(name="samiri2019")
        model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='softmax'))
    elif architecture == ModelArchitecture.MOBILENET:
        model = tf.keras.applications.MobileNetV2(classes=10, weights=None)
    elif architecture == ModelArchitecture.RESNET101:
        model = tf.keras.applications.ResNet101V2(classes=10, weights=None)

    if model and show_summary:
        model.summary()

    return model


def make_datasets_unbatched(datasets, set_name='train', architecture=ModelArchitecture.SA_MIRI):
    # Scaling MNIST data from (0, 255] to (0., 1.]
    def scale(image, label):
        image = tf.cast(image, tf.float32)

        if architecture is not ModelArchitecture.SA_MIRI:
            image = tf.image.resize(image, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        image /= 255

        label = tf.one_hot(label, depth=10)
        return image, label

    if 'train' in set_name:
        return datasets['train'].map(scale, num_parallel_calls=tf.data.experimental.AUTOTUNE).\
            cache().repeat().shuffle(BUFFER_SIZE)
    else:
        return datasets['test'].map(scale, num_parallel_calls=tf.data.experimental.AUTOTUNE)
