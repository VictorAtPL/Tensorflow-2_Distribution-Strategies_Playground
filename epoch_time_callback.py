"""
Callback which prints an average time of epoch execution without the first one and the times of epoch executions.
Tested with: (1) none strategy, (2) OneDeviceStrategy, (3) MirroredStrategy, (4) MultiWorkerMirroredStrategy.
It took me a couple of hours to understand how Variables in tf.distribute works and how to correctly make
a reduction. The most valuable documentation is here:
https://www.tensorflow.org/api_docs/python/tf/distribute/StrategyExtended

I would never try to implement it if I didn't attend to SA-MIRI course at UPC. Thanks for @jorditorresBCN
for possibility to work on tf.distribute and for Barcelona Supercomputing Center that we (students)
had a possibility to use Marenostrum Power9-CTE supercomputer for purposes of this course.

MIT License

Copyright (c) 2019 Piotr Podbielski
"""
import time

import tensorflow as tf
import numpy as np


class EpochTimeCallback(tf.keras.callbacks.Callback):
    epochs_execution_time = None
    epoch_start_time = None
    epoch_time_variable = None

    def on_train_begin(self, logs=None):
        self.epochs_execution_time = []

        with tf.distribute.get_strategy().scope():
            self.epoch_time_variable = tf.Variable(0.0, name="time_diff", dtype=tf.float32)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    @staticmethod
    def f(tensor):
        return tensor

    def on_epoch_end(self, epoch, logs=None):
        time_diff = tf.convert_to_tensor(time.time() - self.epoch_start_time, dtype=tf.float32)
        self.epoch_time_variable.assign(time_diff)

        @tf.function
        def run_experimental_run_v2():
            replica_results = tf.distribute.get_strategy().experimental_run_v2(self.f, args=(self.epoch_time_variable,))
            return replica_results

        replica_result = run_experimental_run_v2()
        reduced_result = tf.distribute.get_strategy().reduce(tf.distribute.ReduceOp.MEAN, replica_result, axis=None)
        self.epochs_execution_time.append(reduced_result.numpy())

    def on_train_end(self, logs=None):
        print("Epoch execution times: {}".format(str(self.epochs_execution_time)))
        print("Epoch execution average time without 1st epoch: {}".format(np.mean(self.epochs_execution_time[1:])))
