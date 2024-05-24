import tensorflow as tf
import os
from glob import glob
import logging
import random
import re

logging.basicConfig(level=logging.DEBUG)

# The following file loading functions assume that you have cropped the images in appropriate patches according to our data_preprocessing steps


def get_datasets(original_images_path, denoised_images_path, batch_size, val_split=0.1):
    original_images = []
    denoised_images = []

    for directory in os.listdir(original_images_path):
        logging.info(f"Loading: {directory}")
        if original_images_path == "/your/path/patches/": # You need to define the folder of your one to one MIDD
            original_images_itter, denoised_images_itter = pair_images_simple(original_images_path+directory+"/original_patches/", denoised_images_path+directory+"/denoised_patches/")
        elif original_images_path == "/your/path/one/to/20/dataset": # You need to define the folder of your full MIDD (20 noisy to 1 GT)
            original_images_itter, denoised_images_itter = pair_images(original_images_path+directory+"/original_20_patches/", denoised_images_path+directory+"/denoised_patches/")
        else:
            original_images = sorted(glob(os.path.join(original_images_path, "*")))
            denoised_images = sorted(glob(os.path.join(denoised_images_path, "*")))
            break
        original_images.append(original_images_itter)
        denoised_images.append(denoised_images_itter)

    original_images = flatten_if_list_of_lists(original_images)
    denoised_images = flatten_if_list_of_lists(denoised_images)

    # Check filenames lists
    index, values = compare_filenames(original_images, denoised_images)

    if index is not None:
        logging.debug(f"The first difference is at index {index}, with values {values[0], values[2]} and {values[1], values[3]} from the two lists.")
    else:
        logging.debug("The noisy and gt image lists are identical.")

    # Shuffle the datasets consistently
    seed = 2345
    random.seed(seed)
    random.shuffle(original_images)
    random.seed(seed)
    random.shuffle(denoised_images)
    
    # Split the chunk into training and validation sets
    num_train_images = int(len(original_images) * (1 - val_split))
    original_images_train = original_images[:num_train_images]
    denoised_images_train = denoised_images[:num_train_images]
    original_images_val = original_images[num_train_images:]
    denoised_images_val = denoised_images[num_train_images:]

    train_dataset = build_dataset(
        original_images= original_images_train,
        denoised_images= denoised_images_train,
        batch_size=batch_size,
    )

    val_dataset = build_dataset(
        original_images= original_images_val,
        denoised_images= denoised_images_val,
        batch_size=batch_size,
    )

    return train_dataset, val_dataset

def pair_images(folder_path_original, folder_path_denoised):
    original_images_out = []
    ground_truth_images = []
    ground_truth_images_out = []
    file_dict = {}

    # Scan through the folder and organize the file names
    for file_name in os.listdir(folder_path_original):
        if "image" in file_name:
            # Extract the base name of the image (without file number and crop info)
            base_name = "_".join(file_name.split("_")[:-4])
            base_name2 = file_name.split("_")[-1][:-4]
            base_name1 = file_name.split("_")[-2]
            base_name = base_name+"_"+base_name1+"_"+base_name2
            
            full_path = os.path.join(folder_path_original, file_name)
            
            # Group the images based on their base name
            if base_name not in file_dict:
                file_dict[base_name] = []
            file_dict[base_name].append(full_path)

    for file_name in os.listdir(folder_path_denoised):
        # Remove file ending
        if file_name[-4:] == ".jpg":
            base_name = file_name[:-4]
        elif file_name[-4:] == "jpeg":
            base_name = file_name[:-5]
        elif file_name[-4:] == ".png":
            base_name = file_name[:-4]
        else:
            logging.info("Wrong file format: ",file_name)
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

def pair_images_simple(folder_path_original, folder_path_denoised):
    original_images_out = []
    ground_truth_images = []
    ground_truth_images_out = []
    file_dict = {}

    for file_name in os.listdir(folder_path_original):
        if file_name[-4:] == ".jpg":
            base_name = file_name[:-4]
        elif file_name[-4:] == "jpeg":
            base_name = file_name[:-5]
        elif file_name[-4:] == ".png":
            base_name = file_name[:-4]
        else:
            logging.info("Wrong file format: ",file_name)
            continue

        full_path = os.path.join(folder_path_original, file_name)
        item = (base_name, full_path)
        
        # Group the images based on their base name
        if base_name not in file_dict:
            file_dict[base_name] = []
        file_dict[base_name].append(full_path)

    # Scan through the folder and organize the file names
    for file_name in os.listdir(folder_path_original):
        if "image" in file_name:
            # Extract the base name of the image (without file number and crop info)
            base_name = "_".join(file_name.split("_")[:-4])
            base_name2 = file_name.split("_")[-1][:-4]
            base_name1 = file_name.split("_")[-2]
            base_name = base_name+"_"+base_name1+"_"+base_name2
            
            # Store the full path of the image file
            full_path = os.path.join(folder_path_original, file_name)
            
            # Group the images based on their base name
            if base_name not in file_dict:
                file_dict[base_name] = []
            file_dict[base_name].append(full_path)

    for file_name in os.listdir(folder_path_denoised):
        if file_name[-4:] == ".jpg":
            base_name = file_name[:-4]
        elif file_name[-4:] == "jpeg":
            base_name = file_name[:-5]
        elif file_name[-4:] == ".png":
            base_name = file_name[:-4]

        full_path = os.path.join(folder_path_denoised, file_name)
        item = (base_name, full_path)

        ground_truth_images.append(item)
            
    # Pair up the noisy and ground truth images
    for gt_base_name, gt_path in ground_truth_images:
        if gt_base_name in file_dict:
            for original_path in file_dict[gt_base_name]:
                original_images_out.append(original_path)
                ground_truth_images_out.append(gt_path)

    return original_images_out, ground_truth_images_out

def build_dataset(original_images, denoised_images, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((original_images, denoised_images))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(int(batch_size), drop_remainder=True)
    return dataset

def load_image(original_path, denoised_path):
    original_image = read_image(original_path)
    denoised_image = read_image(denoised_path)
    return original_image, denoised_image

def augment_image(original_image, denoised_image):
    original_image = tf.image.random_flip_up_down(original_image, seed=4242)
    denoised_image = tf.image.random_flip_up_down(denoised_image, seed=4242)
    original_image = tf.image.random_flip_left_right(original_image, seed=4242)
    denoised_image = tf.image.random_flip_left_right(denoised_image, seed=4242)
    return original_image, denoised_image

def read_image(image_path, normalization_factor=65535.0):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, dtype=tf.uint16)
    image = tf.cast(image, dtype=tf.float32) / normalization_factor
    return image

def compare_filenames(list1, list2):
    for index, (path1, path2) in enumerate(zip(list1, list2)):
        filename1 = os.path.basename(path1)
        filename2 = os.path.basename(path2)

        filename1, _ = os.path.splitext(filename1)
        filename2, _ = os.path.splitext(filename2)

        filename1_matched = remove_file_x_pattern(filename1)

        if filename1_matched != filename2:
            return index, (filename1, filename2, path1, path2)

    return None, None

def remove_file_x_pattern(s):
    # Define the regex pattern to match "file_0" up to "file_19"
    pattern = r'_file_(1?[0-9])'
    result = re.sub(pattern, '', s)
    
    return result

def flatten_if_list_of_lists(lst):
    if all(isinstance(sublist, list) for sublist in lst):
        return [item for sublist in lst for item in sublist]
    return lst