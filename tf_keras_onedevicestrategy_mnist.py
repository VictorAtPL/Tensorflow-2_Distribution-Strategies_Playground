from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from common import get_compiled_model
from epoch_time_callback import EpochTimeCallback

BUFFER_SIZE = 10000


def make_datasets_unbatched(datasets, set_name='train'):
    # Scaling MNIST data from (0, 255] to (0., 1.]
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        label = tf.one_hot(label, depth=10)
        return image, label

    if 'train' in set_name:
        return datasets['train'].map(scale, num_parallel_calls=tf.data.experimental.AUTOTUNE).\
            cache().repeat().shuffle(BUFFER_SIZE)
    else:
        return datasets['test'].map(scale, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def run_training(args):
    strategy = tf.distribute.OneDeviceStrategy("/cpu:0")

    datasets, info = tfds.load(name='mnist',
                               with_info=True,
                               as_supervised=True,
                               shuffle_files=False)

    batch_size = args.batch_size * strategy.num_replicas_in_sync
    learning_rate = args.learning_rate * strategy.num_replicas_in_sync

    with strategy.scope():
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        model = get_compiled_model(learning_rate)

        train_dataset = make_datasets_unbatched(datasets, set_name='train').batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Define the checkpoint directory to store the checkpoints
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                           save_weights_only=True),
        EpochTimeCallback()]

    model.fit(x=train_dataset, epochs=args.epochs,
              steps_per_epoch=info.splits['train'].num_examples // batch_size,
              verbose=2,
              callbacks=callbacks)

    test_dataset = make_datasets_unbatched(datasets, set_name='test').batch(batch_size, drop_remainder=True)
    test_loss, test_acc = model.evaluate(x=test_dataset, verbose=0,
                                         steps=info.splits['test'].num_examples // batch_size)

    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)


def main():
    tfds.disable_progress_bar()

    print("VERSION TF")
    print(tf.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    run_training(parser.parse_args())


if __name__ == '__main__':
    main()
