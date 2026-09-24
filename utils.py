"""Losses, metrics and callbacks used for training and evaluation."""
import os

import keras
import numpy as np
import tensorflow as tf
from keras import ops


def scale_tensor(tensor):
    """Min-max normalise a tensor to [0, 1]."""
    _min, _max = ops.min(tensor), ops.max(tensor)
    return (tensor - _min) / (_max - _min)


def PSNR(original, compressed, max_pixel=255.0):
    """PSNR between two numpy images (100 for identical images)."""
    mse = np.mean((np.asarray(original, np.float64) - np.asarray(compressed, np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def _scaled_psnr(y_true, y_pred, max_val):
    return tf.image.psnr(scale_tensor(y_true), scale_tensor(y_pred), max_val=max_val)


def _scaled_ssim(y_true, y_pred, max_val):
    return tf.image.ssim(scale_tensor(y_true), scale_tensor(y_pred), max_val=max_val)


@keras.saving.register_keras_serializable(package="SplitterNet")
class PSNRMetric(keras.metrics.MeanMetricWrapper):
    """Mean PSNR of min-max normalised images."""

    def __init__(self, max_val=1.0, name="psnr_metric", dtype=None):
        super().__init__(fn=_scaled_psnr, name=name, dtype=dtype, max_val=max_val)
        self.max_val = max_val

    def get_config(self):
        return {"max_val": self.max_val, "name": self.name, "dtype": self.dtype}


@keras.saving.register_keras_serializable(package="SplitterNet")
class SSIMMetric(keras.metrics.MeanMetricWrapper):
    """Mean SSIM of min-max normalised images."""

    def __init__(self, max_val=1.0, name="ssim_metric", dtype=None):
        super().__init__(fn=_scaled_ssim, name=name, dtype=dtype, max_val=max_val)
        self.max_val = max_val

    def get_config(self):
        return {"max_val": self.max_val, "name": self.name, "dtype": self.dtype}


@keras.saving.register_keras_serializable(package="SplitterNet")
class PSNRLoss(keras.losses.Loss):
    """Negative PSNR, so that minimising the loss maximises PSNR."""

    def __init__(self, max_val=1.0, name="psnr_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_val = max_val

    def call(self, y_true, y_pred):
        return -tf.image.psnr(y_true, y_pred, max_val=self.max_val)

    def get_config(self):
        return {**super().get_config(), "max_val": self.max_val}


@keras.saving.register_keras_serializable(package="SplitterNet")
class L1Loss(keras.losses.Loss):
    def __init__(self, name="l1_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return ops.mean(ops.abs(y_true - y_pred))


@keras.saving.register_keras_serializable(package="SplitterNet")
class EDGELoss(keras.losses.Loss):
    """Mean absolute difference between the Sobel edges of prediction and target."""

    def __init__(self, name="edge_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return ops.mean(ops.abs(tf.image.sobel_edges(y_true) - tf.image.sobel_edges(y_pred)))


@keras.saving.register_keras_serializable(package="SplitterNet")
class CharbonnierLoss(keras.losses.Loss):
    def __init__(self, epsilon=1e-3, name="charbonnier_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        return ops.mean(ops.sqrt(ops.square(y_true - y_pred) + self.epsilon ** 2))

    def get_config(self):
        return {**super().get_config(), "epsilon": self.epsilon}


class PrintLearningRate(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(ops.convert_to_numpy(self.model.optimizer.learning_rate))
        print(f"\nLearning rate for epoch {epoch + 1} is {lr:.3e}")


class SaveEveryNSteps(keras.callbacks.Callback):
    """Save the full model every ``save_freq`` training steps."""

    def __init__(self, save_freq, save_path):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.step_count = 0
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        self.step_count += 1
        if self.step_count % self.save_freq == 0:
            filename = f"model_e{self.current_epoch + 1}_step_{self.step_count}.keras"
            self.model.save(os.path.join(self.save_path, filename))
