import numpy as np
import os
import cv2
from joblib import Parallel, delayed

# This code crops the datasset into patches of size=256

def create_patches(filename, images_path, images_patches_path, size=256):
    if not os.path.exists(images_patches_path):
        os.makedirs(images_patches_path)

    try:
        original = cv2.imread(images_path + filename, cv2.IMREAD_UNCHANGED)
        height, width, _ = original.shape
        num_patches_x = width // size
        num_patches_y = height // size

        for i in range(num_patches_x):
            for j in range(num_patches_y):
                x = i * size
                y = j * size
                img1 = original[y:y+size, x:x+size]

                if np.shape(img1) == (size,size,3):
                    cv2.imwrite(images_patches_path+'image_'+filename[:-4]+'_'+str(i)+'_'+str(j)+'.jpg',img1, [cv2.IMWRITE_JPEG_QUALITY, 100])

    except Exception as e:
        # Handle any exception that occurs
        print(f"An error occurred: {e} at {filename}")


# Define folder for image cropping
modality = "denoised" #denoised or original
images_path = "dataset/path/uncropped/" + modality + "/"
images_patches_path = images_path[:-9]+"patches/"+modality+"_patches/"

img_ids = np.asarray(sorted(os.listdir(images_path)))
Parallel(n_jobs=10)(delayed(create_patches)(i, images_path, images_patches_path) for i in sorted(img_ids))


