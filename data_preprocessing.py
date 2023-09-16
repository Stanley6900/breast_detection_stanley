import pandas as pd
import cv2 as cv
import numpy as np
import os
from glob import glob


# define function for function loading.
def data_loading(file_path):
    """ This function loads the csv data.
    Args:
        file_path (str): A string which represents the file path of the file
    Return:
        a pandaframe which represents the loaded file.
    """

    return pd.read_csv(file_path)


def fix_image_path(dataset):
    """
    This function changes the path of both full mammogram, cropped and mask column.
    Args:
        dataset(dataframe): A dataframe containing the columns whose paths are to be fixed.
    """
    for i, img in enumerate(dataset.values):
        img_name = img[11].split("/")[2]
        dataset.iloc[i,11] = full_mammogram_dict[img_name]
        img_name = img[12].split("/")[2]
        dataset.iloc[i,12] = cropped_dict[img_name]
        img_name = img[13].split("/")[2]
        dataset.iloc[i,13] = masked_dict[img_name]
        
        
# create new directories.
def create_new_directory(direc_path):
    """ This function creates new directory.
    Args:
        direct_name(str): A string which represents the directory name.
    """
    # create if it doesn't already exists.
    os.makedirs(direc_path, exist_ok=True)
    # Check if the directory has been created
    if os.path.exists(direc_path):
        print(f"The directory '{direc_path}' has been created or already exists.")
    else:
        print(f"Failed to create the directory '{direc_path}'.")


def create_save_full_images(dataframe, output_directory, type):
    """
    Create a new directory, delete existing files if any, and save images with new names based on a DataFrame.
    """
    try:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        else:
            remove_existing_files(output_directory)

        for index, row in dataframe.iterrows():
            image_path = row['image file path']  # Assuming you have a 'image_path' column in your DataFrame
            img = cv.imread(image_path, cv.IMREAD_COLOR)
            new_image_name = f"image_{type}{index + 1}.jpg"
            save_image(img, os.path.join(output_directory, new_image_name))

    except OSError as error:
        print(f"Directory '{output_directory}' could not be created: {error}")


def create_save_mask_images(dataframe, output_directory, type):
    """
    Create a new directory, delete existing files if any, and save images with new names based on a DataFrame.
    """
    try:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        else:
            remove_existing_files(output_directory)

        for index, row in dataframe.iterrows():
            image_path = row['ROI mask file path']  # Assuming you have a 'image_path' column in your DataFrame
            img = cv.imread(image_path, cv.IMREAD_COLOR)
            new_image_name = f"mask_{type}{index + 1}.jpg"
            save_image(img, os.path.join(output_directory, new_image_name))

    except OSError as error:
        print(f"Directory '{output_directory}' could not be created: {error}")

def create_save_cropped_images(dataframe, output_directory, type):
    """
    Create a new directory, delete existing files if any, and save images with new names based on a DataFrame.
    """
    try:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        else:
            remove_existing_files(output_directory)

        for index, row in dataframe.iterrows():
            image_path = row['cropped image file path']  # Assuming you have a 'image_path' column in your DataFrame
            img = cv.imread(image_path, cv.IMREAD_COLOR)
            new_image_name = f"cropped_{type}{index + 1}.jpg"
            save_image(img, os.path.join(output_directory, new_image_name))

    except OSError as error:
        print(f"Directory '{output_directory}' could not be created: {error}")

def remove_existing_files(directory):
    """
    Remove all files in the specified directory.
    """
    try:
        for file_path in glob(os.path.join(directory, "*")):
            os.remove(file_path)
    except OSError as error:
        print(f"Error while removing files from '{directory}': {error}")


def save_image(image, output_path):
    """
    Save the input image to the specified output path.
    """
    cv.imwrite(output_path, cv.cvtColor(image, cv.COLOR_BGR2RGB))