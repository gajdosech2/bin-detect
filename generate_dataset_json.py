import argparse
import os
import json
import numpy as np


def get_transforms(transform):
    """
    Fixes bad transform from det(R) = -1 to det(R) = 1
    :param transform: original transform
    :return: original transform, fixed transformation
    """
    orig_transform = transform
    proper_transform = np.copy(transform)
    if np.linalg.det(orig_transform) < 0:
        proper_transform[:, 1] *= -1

    return orig_transform, proper_transform


def generate_dataset(path, ignore_bad=True, ignore_bad_det=False):
    """
    Generates json from annotated data. JSON filename depends on the params.
    If annots with det(R) = -1 the second column is flipped for correct representation.

    :param path: path to dataset root folder. JSON file is saved here as well.
    :param ignore_bad: whether to ignore 'bad', 'badish', 'catastrophic' etc. files
    :param ignore_bad_det: whether to ignore annotations where det(R) = -1
    :return:
    """
    dirs = [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]

    entries = []

    for dir in dirs:
        if ignore_bad:
            good_txt_files = [f for f in os.listdir(os.path.join(path, dir)) if
                              '.txt' in f
                              and not any(sub in f for sub in ['bad', 'catas', 'ish', 'pred', 'icp'])]
        else:
            good_txt_files = [f for f in os.listdir(os.path.join(path, dir)) if '.txt' in f
                              and 'pred' not in f
                              and 'icp' not in f]
        for txt_file in good_txt_files:
            txt_path = os.path.join(path, dir, txt_file)
            transform = np.loadtxt(txt_path, max_rows=1).reshape([4, 4]).T
            if ignore_bad_det and np.linalg.det(transform) < 0.0:
                continue
            orig_transform, proper_transform = get_transforms(transform)
            try:
                corners = np.loadtxt(txt_path, skiprows=2, delimiter=',')
            except:
                corners = np.array([])
            name_path = txt_file.split('.')[0]
            exr_normals_path = os.path.join(dir, '{}_normals.exr'.format(name_path))
            exr_positions_path = os.path.join(dir, '{}_positions.exr'.format(name_path))

            entry = {'dir': dir, 'exr_normals_path': exr_normals_path, 'exr_positions_path': exr_positions_path,
                     'txt_path': os.path.join(dir, txt_file), 'corners': corners.tolist(),
                     'orig_transform': transform.tolist(), 'proper_transform': proper_transform.tolist()}

            entries.append(entry)

    if ignore_bad_det:
        json_name = 'dataset_posdet.json' if ignore_bad else 'dataset_all_posdet.json'
    else:
        json_name = 'dataset.json' if ignore_bad else 'dataset_all.json'
    json_path = os.path.join(path, json_name)

    print("Dataset contains {} entries".format(len(entries)))
    print("Saving to: ", json_path)

    with open(json_path, 'w') as f:
        json.dump(entries, f, indent=None)


if __name__ == '__main__':
    """
    Generates four json files for dataset given in path
    Example usage: python generate_dataset.py /path/to/MLBinsDataset/EXR
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to dataset root folder.')
    args = parser.parse_args()
    path = args.path

    generate_dataset(path, ignore_bad=True)
    generate_dataset(path, ignore_bad=False)
    generate_dataset(path, ignore_bad=True, ignore_bad_det=True)
    generate_dataset(path, ignore_bad=False, ignore_bad_det=True)
