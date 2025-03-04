from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from common import get_model, make_datasets_unbatched, ModelArchitecture
from epoch_time_callback import EpochTimeCallback


def run_training(args):
    strategy = tf.distribute.OneDeviceStrategy("/cpu:0")

    datasets, info = tfds.load(name='mnist' if args.architecture == ModelArchitecture.SA_MIRI else 'cifar10',
                               with_info=True,
                               as_supervised=True,
                               shuffle_files=False)

    batch_size = args.batch_size * strategy.num_replicas_in_sync
    learning_rate = args.learning_rate * strategy.num_replicas_in_sync

    with strategy.scope():
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        model = get_model(architecture=args.architecture)
        opt = tf.keras.optimizers.SGD(learning_rate)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

        train_dataset = make_datasets_unbatched(datasets, set_name='train', architecture=args.architecture).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    checkpoint_dir = '/gpfs/projects/sam14/sam14016/training_checkpoints/{}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                           save_weights_only=True,
                                           save_best_only=True,
                                           monitor='loss'),
        EpochTimeCallback()]

    model.fit(x=train_dataset, epochs=args.epochs,
              steps_per_epoch=info.splits['train'].num_examples // batch_size,
              verbose=2,
              callbacks=callbacks)

    test_dataset = make_datasets_unbatched(datasets, set_name='test', architecture=args.architecture).batch(batch_size, drop_remainder=True)
    test_loss, test_acc = model.evaluate(x=test_dataset, verbose=0,
                                         steps=info.splits['test'].num_examples // batch_size)

    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)


def main():
    tfds.disable_progress_bar()

    print("VERSION TF")
    print(tf.__version__)
    if "SLURM_CPUS_PER_TASK" in os.environ and "SLURM_JOB_NODELIST" in os.environ:
        print("SLURM_CPUS_PER_TASK: {}\nSLURM_JOB_NODELIST: {}".format(os.environ['SLURM_CPUS_PER_TASK'],
                                                                       os.environ['SLURM_JOB_NODELIST']))

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=ModelArchitecture.argparse, choices=list(ModelArchitecture), required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    run_training(parser.parse_args())


if __name__ == '__main__':
    main()
