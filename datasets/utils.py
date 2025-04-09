import numpy as np
import nibabel as nib


def split_left_right(file_name, output_path):
    nifti_img = nib.load(file_name)
    segmentation_data = nifti_img.get_fdata()

    LABEL = 2
    LEFT_LABEL = 5
    RIGHT_LABEL = 6

    kidney_indices = np.argwhere(segmentation_data == LABEL)
    new_segmentation_data = np.copy(segmentation_data)

    for idx in kidney_indices:
        x, y, z = idx
        # determine left or right based on x axis
        if x < segmentation_data.shape[0] // 2:
            new_segmentation_data[x, y, z] = LEFT_LABEL
        else:
            new_segmentation_data[x, y, z] = RIGHT_LABEL

    new_nifti_img = nib.Nifti1Image(new_segmentation_data, nifti_img.affine)
    nib.save(new_nifti_img, output_path)


def generate_point_prompt(mask):
    """
    :param mask:
    :return: (i0, j0)
    """
    rows, cols = np.where(mask)
    x_min, x_max = cols.min(), cols.max()
    y_min, y_max = rows.min(), rows.max()
    x = x_max - x_min
    y = y_max - y_min

    while True:
        i0, j0 = np.random.choice(rows), np.random.choice(cols)
        if (i0 + int(0.1 * x) < mask.shape[1] and mask[i0, j0 + int(0.1 * x)] and
                i0 - int(0.1 * x) >= 0 and mask[i0, j0 - int(0.1 * x)] and
                j0 + int(0.1 * y) < mask.shape[0] and mask[j0 + int(0.1 * y), i0] and
                j0 - int(0.1 * y) >= 0 and mask[j0 - int(0.1 * y), i0]):
            break
    return (i0, j0)


def generate_bbox_prompt(mask):
    """
    :param mask:
    :return: (x1, y1, x2, y2)
    """
    rows, cols = np.where(mask)
    i1 = int(np.mean(rows))
    j1 = int(np.mean(cols))

    x_min, x_max = cols.min(), cols.max()
    y_min, y_max = rows.min(), rows.max()
    x = x_max - x_min
    y = y_max - y_min

    t1 = np.random.uniform(0.1, 0.3)
    t2 = np.random.uniform(0.1, 0.3)

    x1 = int(i1 - t1 * x)
    y1 = int(j1 - t2 * y)
    x2 = int(i1 + t1 * x)
    y2 = int(j1 + t2 * y)

    return (x1, y1, x2, y2)
