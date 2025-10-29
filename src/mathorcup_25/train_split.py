"""
Split between train and val folders

This script splits a dataset into training and validation folders.
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path


def parse_arguments():
    """Parse and return user input arguments."""
    parser = argparse.ArgumentParser(
        description="Split dataset into train and validation folders"
    )
    parser.add_argument(
        "--datapath",
        help="Path to data folder containing image and annotation files",
        required=True,
    )
    parser.add_argument(
        "--train_pct",
        help='Ratio of images to go to train folder; the rest go to validation folder (example: ".8")',
        default=0.8,
    )
    return parser.parse_args()


def validate_arguments(data_path, train_percent):
    """Validate user input arguments."""
    if not os.path.isdir(data_path):
        print(
            "Directory specified by --datapath not found. Verify the path is correct (and uses double back slashes if on Windows) and try again."
        )
        sys.exit(0)
    if train_percent < 0.01 or train_percent > 0.99:
        print("Invalid entry for train_pct. Please enter a number between .01 and .99.")
        sys.exit(0)


def create_directories(train_img_path, train_txt_path, val_img_path, val_txt_path):
    """Create output folders if they don't already exist."""
    for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created folder at {dir_path}.")


def get_file_lists(input_image_path, input_label_path):
    """Get list of all images and annotation files."""
    img_file_list = [path for path in Path(input_image_path).rglob("*")]
    txt_file_list = [path for path in Path(input_label_path).rglob("*")]

    print(f"Number of image files: {len(img_file_list)}")
    print(f"Number of annotation files: {len(txt_file_list)}")

    return img_file_list, txt_file_list


def calculate_split_counts(file_num, train_percent):
    """Calculate number of files for train and validation sets."""
    train_num = int(file_num * train_percent)
    val_num = file_num - train_num

    print(f"Images moving to train: {train_num}")
    print(f"Images moving to validation: {val_num}")

    return train_num, val_num


def split_and_copy_files(
    img_file_list,
    input_label_path,
    train_num,
    val_num,
    train_img_path,
    train_txt_path,
    val_img_path,
    val_txt_path,
):
    """Split files and copy them to train or validation folders."""
    for i, set_num in enumerate([train_num, val_num]):
        for ii in range(set_num):
            img_path = random.choice(img_file_list)
            img_fn = img_path.name
            base_fn = img_path.stem
            txt_fn = base_fn + ".txt"
            txt_path = os.path.join(input_label_path, txt_fn)

            # Determine destination folders
            if i == 0:  # Copy first set of files to train folders
                new_img_path, new_txt_path = train_img_path, train_txt_path
            elif i == 1:  # Copy second set of files to the validation folders
                new_img_path, new_txt_path = val_img_path, val_txt_path

            # Copy files
            shutil.copy(img_path, os.path.join(new_img_path, img_fn))
            if os.path.exists(
                txt_path
            ):  # If txt path does not exist, this is a background image, so skip txt file
                shutil.copy(txt_path, os.path.join(new_txt_path, txt_fn))

            img_file_list.remove(img_path)


def main():
    """Main function to execute dataset splitting."""
    # Parse and validate arguments
    args = parse_arguments()
    data_path = args.datapath
    train_percent = float(args.train_pct)

    validate_arguments(data_path, train_percent)

    # Define paths
    input_image_path = os.path.join(data_path, "images")
    input_label_path = os.path.join(data_path, "labels")

    # data_path = os.getdata_path()
    train_img_path = os.path.join(data_path, "train/images")
    train_txt_path = os.path.join(data_path, "train/labels")
    val_img_path = os.path.join(data_path, "validation/images")
    val_txt_path = os.path.join(data_path, "validation/labels")

    # Create folders
    create_directories(train_img_path, train_txt_path, val_img_path, val_txt_path)

    # Get file lists
    img_file_list, txt_file_list = get_file_lists(input_image_path, input_label_path)

    # Calculate split counts
    file_num = len(img_file_list)
    train_num, val_num = calculate_split_counts(file_num, train_percent)

    # Split and copy files
    split_and_copy_files(
        img_file_list,
        input_label_path,
        train_num,
        val_num,
        train_img_path,
        train_txt_path,
        val_img_path,
        val_txt_path,
    )

    print("Dataset split completed successfully!")


if __name__ == "__main__":
    main()
