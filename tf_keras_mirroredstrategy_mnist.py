from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import time

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import models, layers

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
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=
        tf.distribute.NcclAllReduce(),
        # tf.distribute.HierarchicalCopyAllReduce(),
        # tf.distribute.ReductionToOneDevice(),
    )

    datasets, info = tfds.load(name='mnist',
                               with_info=True,
                               as_supervised=True,
                               shuffle_files=False)

    batch_size = args.batch_size * strategy.num_replicas_in_sync
    learning_rate = args.learning_rate * strategy.num_replicas_in_sync

    with strategy.scope():
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        model = models.Sequential()
        model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='softmax'))
        model.summary()

        opt = tf.keras.optimizers.SGD(learning_rate)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

        train_dataset = make_datasets_unbatched(datasets, set_name='train').batch(batch_size)

    test_dataset = make_datasets_unbatched(datasets, set_name='test').batch(batch_size, drop_remainder=True)

    # Define the checkpoint directory to store the checkpoints
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                           save_weights_only=True)]

    model.fit(x=train_dataset, epochs=args.epochs,
              steps_per_epoch=info.splits['train'].num_examples // batch_size,
              verbose=2,
              callbacks=callbacks)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
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

    start_time = time.time()
    run_training(parser.parse_args())
    elapsed_time = time.time() - start_time
    print('Time: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


if __name__ == '__main__':
    main()
