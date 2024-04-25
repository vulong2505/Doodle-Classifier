""" Preprocess raw data. """

##### Libraries

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# Directory Paths
# =============================================================================

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(CURRENT_SCRIPT_PATH))
RAW_DATA_PATH = os.path.join(ROOT_PATH, 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(ROOT_PATH, 'data', 'processed')

# =============================================================================
# Data Handling Functions
# =============================================================================

def load_data(file_path):
    """ Load numpy bitmaps of current .npy file and return numpy bitmap """

    return np.load(file_path)

def normalize_data(data):
    """ Normalize to range [0, 1]"""

    return (data / 255.0).astype(np.float32)

def random_perturbations(data):
    """ Apply random pertubations (angle, scale, rotation, translation, noise) to every image in the data. """

    perturbated_data = []
    for image in data:
        # Reshape flattened image into 28x28
        image_2d = image.reshape(28, 28)
        rows, cols = image_2d.shape

        # Random rotation between -1 to 1 degrees
        angle = np.random.uniform(-10, 10)
        M_rotate = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(image_2d, M_rotate, (cols, rows))  # cv2.warpAffine expects dimensions as (width, height), so I pass (cols, rows)

        # Random scale between 90% to 110%
        scale = np.random.uniform(0.9, 1.1)
        M_scale = cv2.getRotationMatrix2D((cols/2, rows/2), 0, scale)
        scaled = cv2.warpAffine(rotated, M_scale, (cols, rows))

        # Random translation by offsetting x and y between -3 and 3 pixels
        x_offset = np.random.randint(-3, 3)
        y_offset = np.random.randint(-3, 3)
        M_translate = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        translated = cv2.warpAffine(scaled, M_translate, (cols, rows))

        # Random noise
        noise_prob = np.random.uniform(0.01, 0.08)
        noise_mask = np.random.binomial(1, noise_prob, (rows, cols))
        noise_values = np.random.uniform(0, 0.5, (rows, cols))
        noise = np.multiply(noise_mask, noise_values).astype(np.float32)
        noised = cv2.add(translated, noise)

        # Flatten image again and add to perturbed data
        image_flatten = noised.reshape(28*28, 1)
        perturbated_data.append(image_flatten)
    
    return np.array(perturbated_data)

# =============================================================================
# Preprocess Data
# =============================================================================

if __name__ == '__main__':
    """ Process all raw bitmap files in data/raw and saves them in data/processed. """

    # All numpy bitmap files
    raw_files = os.listdir(RAW_DATA_PATH)

    # Get min dataset length
    bm_dataset_length = []
    for raw_file in raw_files:
        bm_data = np.load(os.path.join(PROCESSED_DATA_PATH, raw_file))
        bm_dataset_length.append(len(bm_data))
    min_dataset_length = min(bm_dataset_length)

    for raw_file in raw_files:
        # Get data from a raw file
        bm_data = np.load(os.path.join(RAW_DATA_PATH, raw_file))   

        # Normalize all data to range [0, 1]
        bm_data = normalize_data(bm_data)            

        # Add random angle, scale, offset (x, y), noise for slightly perturbed images to be used in training               
        bm_data = random_perturbations(bm_data)   

        # Cutdown dataset size to balance out sizes for each category
        bm_data = bm_data[:min_dataset_length]                  

        # Save file
        np.save(os.path.join(PROCESSED_DATA_PATH, raw_file), bm_data)


