import tensorflow as tf
import os
import numpy as np
from math import log10, sqrt

class SaveEveryNSteps(tf.keras.callbacks.Callback):
    def __init__(self, save_freq, save_path):
        super(SaveEveryNSteps, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.step_count = 0
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        self.step_count += 1
        if self.step_count % self.save_freq == 0:
            filename = f"model_e{self.current_epoch + 1}_step_{self.step_count}.h5"
            self.model.save(os.path.join(self.save_path, filename))


class PSNRMetric(tf.keras.metrics.Metric):
    def __init__(self, max_val: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_val = max_val
        self.psnr = tf.keras.metrics.Mean(name="psnr")

    def update_state(self, y_true, y_pred, *args, **kwargs):
        psnr = tf.image.psnr(
            scale_tensor(y_true), scale_tensor(y_pred), max_val=self.max_val
        )
        self.psnr.update_state(psnr, *args, **kwargs)

    def result(self):
        return self.psnr.result()

    def reset_state(self):
        self.psnr.reset_state()


class SSIMMetric(tf.keras.metrics.Metric):
    def __init__(self, max_val: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_val = max_val
        self.ssim = tf.keras.metrics.Mean(name="ssim")

    def update_state(self, y_true, y_pred, *args, **kwargs):
        ssim = tf.image.ssim(
            scale_tensor(y_true), scale_tensor(y_pred), max_val=self.max_val
        )
        self.ssim.update_state(ssim, *args, **kwargs)

    def result(self):
        return self.ssim.result()

    def reset_state(self):
        self.ssim.reset_state()


class PSNRLoss(tf.keras.losses.Loss):
    def __init__(self, max_val: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_val = max_val

    def call(self, y_true, y_pred):
        return -tf.image.psnr(y_true, y_pred, max_val=self.max_val)
    
class L1Loss(tf.keras.losses.Loss):
    def __init__(self, max_val: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_val = max_val

    def call(self, y_true, y_pred):
        return self.l1_loss(y_true, y_pred)
    
    def l1_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))

class EDGELoss(tf.keras.losses.Loss):
    def __init__(self, max_val: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_val = max_val

    def call(self, y_true, y_pred):
        return self.edge_loss(y_true, y_pred)

    def edge_loss(self, y_true, y_pred):
        # Compute the gradient of the predicted output
        gradient_y_pred = tf.image.sobel_edges(y_pred)
        
        # Compute the gradient of the ground truth output
        gradient_y_true = tf.image.sobel_edges(y_true)
        
        # Compute the mean absolute difference between the gradients
        loss = tf.reduce_mean(tf.abs(gradient_y_true - gradient_y_pred))
        
        return loss
    
class PrintLearningRate(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        learning_rate = tf.keras.backend.get_value(optimizer.learning_rate)
        print(f"\nLearning rate for epoch {epoch+1} is {learning_rate}")

class CharbonnierLoss(tf.keras.losses.Loss):
    def __init__(self, epsilon: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = tf.convert_to_tensor(epsilon)

    def call(self, y_true, y_pred):
        squared_difference = tf.square(y_true - y_pred)
        return tf.reduce_mean(tf.sqrt(squared_difference + tf.square(self.epsilon)))


def scale_tensor(tensor):
    _min = tf.math.reduce_min(tensor)
    _max = tf.math.reduce_max(tensor)
    return (tensor - _min) / (_max - _min)

def PSNR(original, compressed):
    mse = np.mean((original-compressed) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20*log10(max_pixel/sqrt(mse))
    return psnr

def pair_images(folder_path_original, folder_path_denoised):
    original_images_out = []
    ground_truth_images = []
    ground_truth_images_out = []
    file_dict = {}

    # Scan through the folder and organize the file names
    for file_name in os.listdir(folder_path_original):
        # This is assumed to be a ground truth image
        if file_name[-4:] == ".jpg":
            base_name = file_name[:-4]
        elif file_name[-4:] == "jpeg":
            base_name = file_name[:-5]
        elif file_name[-4:] == ".png":
            base_name = file_name[:-4]
        elif file_name[-4:] == ".PNG":
            base_name = file_name[:-4]
        else:
            continue

        full_path = os.path.join(folder_path_original, file_name)
        item = (base_name, full_path)

        if base_name not in file_dict:
            file_dict[base_name] = []
        file_dict[base_name].append(full_path)


    for file_name in os.listdir(folder_path_denoised):
        # This is assumed to be a ground truth image
        if file_name[-4:] == ".jpg":
            base_name = file_name[:-4]
        elif file_name[-4:] == "jpeg":
            base_name = file_name[:-5]
        elif file_name[-4:] == ".png":
            base_name = file_name[:-4]
        elif file_name[-4:] == ".PNG":
            base_name = file_name[:-4]
        else:
            continue
        full_path = os.path.join(folder_path_denoised, file_name)
        item = (base_name, full_path)

        ground_truth_images.append(item)
            
    # Pair up the noisy and ground truth image
    for gt_base_name, gt_path in ground_truth_images:
        if gt_base_name in file_dict:
            for original_path in file_dict[gt_base_name]:
                original_images_out.append(original_path)
                ground_truth_images_out.append(gt_path)

    return original_images_out, ground_truth_images_out
