import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import gc
import logging
from glob import glob
from utils import pair_images, PSNR

def run(saved_model_path, mode, dataset, create_model=False, model=None):
    logging.basicConfig(level=logging.DEBUG)

    if mode == "evaluate_saved_model":
        evaluate_saved_model(saved_model_path, dataset)

def evaluate_saved_model(saved_model_path, test_set_dir):
    logging.basicConfig(level=logging.DEBUG)

    if isinstance(saved_model_path, str):
        model = tf.keras.models.load_model(saved_model_path, compile=False)
        model.compile()
    else:
        model = saved_model_path
    model.summary()

    original_images = []
    denoised_images = []

    if test_set_dir == "/your/path/test_set/":
        for directory in os.listdir(test_set_dir):
            original_images_itter, denoised_images_itter = pair_images(test_set_dir+directory+"/test_set/original/", test_set_dir+directory+"/test_set/denoised/")
            original_images.append(original_images_itter)
            denoised_images.append(denoised_images_itter)
        original_images = [item for sublist in original_images for item in sublist]
        denoised_images = [item for sublist in denoised_images for item in sublist]
    else:
        original_images.extend(sorted(glob(os.path.join(test_set_dir+"/original/", "*"))))
        denoised_images.extend(sorted(glob(os.path.join(test_set_dir+"/denoised/", "*"))))

    snr_values_denoised = []
    snr_values_original = []
    ssim_values_denoised = []
    ssim_values_original = []

    logging.debug(f"Length: {len(original_images)}")

    # Iterate over the files and print their names
    for i, file_name in enumerate(original_images):
        if 'blurry' in file_name:
            continue
        try:
            denoised_file_name = denoised_images[i]
            logging.debug(f"File: {file_name}")
            logging.debug(f"File: {denoised_file_name}")

            image_gt = tf.io.read_file(denoised_file_name)
            image_gt = tf.io.decode_png(image_gt, channels=3)
            image_gt = tf.cast(image_gt, dtype=tf.float32) / 255.0

            noisy_image = tf.io.read_file(file_name)
            noisy_image = tf.io.decode_png(noisy_image, channels=3)
            noisy_image = tf.cast(noisy_image, dtype=tf.float32) / 255.0
            noisy_image = np.expand_dims(noisy_image, axis=0)

            denoised_image = model.predict(noisy_image)
            denoised_image = np.squeeze(denoised_image)

            value_psnr_noisy = PSNR(np.array(image_gt*255.0), np.array(noisy_image*255.0))
            value_psnr_denoised = PSNR(np.array(image_gt*255.0), np.array(denoised_image*255.0))

            value_ssim_original = tf.image.ssim(np.array(image_gt*255), np.array(noisy_image*255),255.0,filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03,return_index_map=False).numpy()[0]
            value_ssim_denoised = tf.image.ssim(np.array(image_gt*255), np.array(denoised_image*255),255.0,filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03,return_index_map=False).numpy()
            
            ssim_values_original = np.append(ssim_values_original, value_ssim_original)
            ssim_values_denoised = np.append(ssim_values_denoised, value_ssim_denoised)
            
            snr_values_original = np.append(snr_values_original, value_psnr_noisy)
            snr_values_denoised = np.append(snr_values_denoised, value_psnr_denoised)
            logging.debug(f"Difference: {value_psnr_denoised-value_psnr_noisy}")
            logging.debug(f"Value_ssim_original: {value_ssim_original}, Value_ssim_denoised: {value_ssim_denoised}")
            logging.debug(f"Value_psnr_noisy: {value_psnr_noisy}, Value_psnr_denoised: {value_psnr_denoised}")
            gc.collect()

        except Exception as e:
            logging.info(f"An error occurred: {e}")

    avg_denoised_psnr=np.mean(snr_values_denoised)
    avg_denoised_ssim=np.mean(ssim_values_denoised)

    avg_orig_psnr=np.mean(snr_values_original)
    avg_orig_ssim=np.mean(ssim_values_original)
    logging.info(f"avg_denoised_psnr: {avg_denoised_psnr}, avg_orig_psnr: {avg_orig_psnr}")
    logging.info(f"avg_denoised_ssim: {avg_denoised_ssim}, avg_orig_ssim: {avg_orig_ssim}")

    del model
    tf.keras.backend.clear_session()
    
    return avg_denoised_psnr, avg_denoised_ssim
