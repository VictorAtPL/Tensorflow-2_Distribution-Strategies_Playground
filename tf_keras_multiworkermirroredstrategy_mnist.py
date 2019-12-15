from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from common import get_model, make_datasets_unbatched
from epoch_time_callback import EpochTimeCallback

BUFFER_SIZE = 10000


def run_training(args):
    nodes_number = int(os.environ['SLURM_NTASKS'])
    resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(jobs={"worker": nodes_number},
                                                                   gpus_per_node=4, gpus_per_task=4)

    cluster_spec_dict = resolver.cluster_spec().as_dict()
    task_type, task_id = resolver.get_task_info()

    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': cluster_spec_dict,
        'task': {'type': 'worker', 'index': task_id}
    })

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        communication=
        tf.distribute.experimental.CollectiveCommunication.RING,
        # tf.distribute.experimental.CollectiveCommunication.NCCL # not working right now
    )

    datasets, info = tfds.load(name='mnist',
                               with_info=True,
                               as_supervised=True,
                               shuffle_files=False)

    batch_size = args.batch_size * strategy.num_replicas_in_sync
    learning_rate = args.learning_rate * strategy.num_replicas_in_sync

    with strategy.scope():
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        model = get_model()
        opt = tf.keras.optimizers.SGD(learning_rate)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

        train_dataset = make_datasets_unbatched(datasets, set_name='train').batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

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
              verbose=2 if task_id == 0 else 0,
              callbacks=callbacks)

    if task_id != 0:
        return

    os.environ.pop('TF_CONFIG')
    model = get_model(show_summary=False)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    batch_size = args.batch_size

    test_dataset = make_datasets_unbatched(datasets, set_name='test').batch(batch_size, drop_remainder=True)
    test_loss, test_acc = model.evaluate(x=test_dataset, verbose=0,
                                         steps=info.splits['test'].num_examples // batch_size)

    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)


def main():
    tfds.disable_progress_bar()

    print("VERSION TF")
    print(tf.__version__)
    print("SLURM_CPUS_PER_TASK: {}\nSLURM_JOB_NODELIST: {}".format(os.environ['SLURM_CPUS_PER_TASK'], os.environ['SLURM_JOB_NODELIST']))

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    run_training(parser.parse_args())


if __name__ == '__main__':
    main()
