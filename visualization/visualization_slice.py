import os

import matplotlib.pyplot as plt
import nibabel as nib
import nrrd
import numpy as np
from matplotlib.colors import ListedColormap

from visualization_config import get_mask_colors


def get_slice_image(image_path, label_path, label_slice_index, image_slice_index):
    image_nii = nib.load(image_path)
    image_data = image_nii.get_fdata()
    if label_path.endswith("nii.gz"):
        label_nii = nib.load(label_path)
        label_data = label_nii.get_fdata()
    else:
        label_data, _ = nrrd.read(label_path)

    unique_labels = set(label_data.flatten())
    label_count = len(unique_labels)
    print(f"Label Count : {label_count}")

    # get slice
    image_slice = image_data[:, :, image_slice_index]
    label_slice = label_data[:, :, label_slice_index]
    image_slice = np.rot90(image_slice, k=1)
    label_slice = np.rot90(label_slice, k=1)
    return image_slice, label_slice


def download_image(dataset_name, image_slice, label_slice, output_path):
    custom_colors = get_mask_colors(dataset_name)
    color_count = len(custom_colors)
    print(f"Color Count : {color_count}")
    # custom_colors = [(0, 0, 0, 0)] + custom_colors

    unique_labels = np.unique(label_slice)
    unique_labels = unique_labels.astype(int)
    unique_labels = unique_labels[unique_labels != 0]
    # print(unique_labels)

    actual_colors = [custom_colors[i - 1] for i in unique_labels]
    cmap = ListedColormap(actual_colors)

    new_label_slice = np.zeros_like(label_slice)
    for i, label in enumerate(unique_labels):
        new_label_slice[label_slice == label] = i + 1

    fig, ax = plt.subplots()
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.imshow(image_slice, cmap='gray', interpolation='bilinear')
    new_label_slice = np.ma.masked_where(new_label_slice == 0, new_label_slice)
    plt.imshow(new_label_slice, cmap=cmap, alpha=0.75)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()


if __name__ == '__main__':
    dataset_name = "AbdomenCT-1k"
    image_path = '../input/AbdomenCT-1k/Case_00888_0000.nii.gz'
    label_path = '../input/AbdomenCT-1k/Case_00888_split_no.nii.gz.seg.nrrd'
    if label_path.endswith("seg.nrrd"):
        model_name = label_path.split("/")[-1].split(".")[0].split("_")[-1]
    else:
        model_name = "GT"
    output_name = image_path.split('/')[3].split('.')[0]
    output_path = os.path.join('../output/AbdomenCT-1k/', output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    png_path = os.path.join(output_path, model_name + ".png")
    print(f"output path : {png_path}")
    # 409 61/72
    # 888 63
    label_slice_index = 67
    image_slice_index = 67

    image_slice, label_slice = get_slice_image(image_path=image_path, label_path=label_path,
                                               image_slice_index=image_slice_index, label_slice_index=label_slice_index)
    download_image(dataset_name, image_slice, label_slice, png_path)
