import os
import json
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


def get_canonical_transform(transform):
    """
    Unused - Takes rotation matrix and finds canonical representation w.r.t. symmetries as per:
    https://arxiv.org/pdf/1908.07640.pdf check eq (22) for this case specifically
    """

    rot = transform[:3, :3]

    # we need to consider only one symmetry e.g. 180 deg around z axis
    sym_rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    if np.linalg.norm(sym_rot @ rot - np.eye(3), ord='fro') < np.linalg.norm(rot - np.eye(3), ord='fro'):
        sym_rot_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        regressor = 1
        if np.linalg.norm(sym_rot @ rot - sym_rot_90, ord='fro') < np.linalg.norm(rot - sym_rot_90, ord='fro'):
            rot = sym_rot @ rot
    else:
        regressor = 0

    transform[:3, :3] = rot
    return transform, np.array([regressor], dtype=np.float32)


class Dataset(Dataset):
    def __init__(self, path, split, width, height, preload=True, noise_sigma=None, t_sigma=0.0, random_rot=False):
        self.dataset_dir = os.path.dirname(path)
        self.split = split
        self.width = width
        self.height = height
        self.preload = preload
        self.noise_sigma = noise_sigma
        self.t_sigma = t_sigma
        self.random_rot = random_rot

        print("Loading dataset from path: ", path)
        with open(path, 'r') as f:
            self.entries = json.load(f)

        if 'train' not in path and 'val' not in path:
            if self.split == 'train':
                self.entries = [entry for i, entry in enumerate(self.entries) if i % 5 != 0]
            elif self.split == 'val':
                self.entries = [entry for i, entry in enumerate(self.entries) if i % 5 == 0]

        print("Split: ", self.split)
        print("Size: ", len(self))
        if self.preload:
            print("Preloading exrs to memory")
            for entry in self.entries:
                print(entry)
                entry['xyz'] = self.load_xyz(entry)

    def __len__(self):
        """
        Length of dataset
        :return: number of elements in dataset
        """
        return len(self.entries)

    def load_xyz(self, entry):
        """
        Loads pointcloud for a given entry
        :param entry: entry from self.entries
        :return: pointcloud wit shape (3, height, width)
        """
        exr_path = os.path.join(self.dataset_dir, entry['exr_positions_path'])
        xyz = cv2.imread(exr_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if xyz is None:
            print(exr_path)
            raise ValueError("Image at path ", exr_path)
        xyz = cv2.resize(xyz, (self.width, self.height), cv2.INTER_NEAREST_EXACT)
        xyz = np.transpose(xyz, [2, 0, 1])
        return xyz

    def get_aug_transform(self):
        """
        Generates random transformation using. R is from SO(3) thanks to QR decomposition.
        :return: random transformation matrix
        """
        if self.random_rot:
            R, _ = np.linalg.qr(np.random.randn(3, 3))
        else:
            R = np.eye(3)

        t = self.t_sigma * np.random.randn(3)

        out = np.zeros([4, 4])
        out[:3, :3] = R
        out[:3, 3] = t
        out[3, 3] = 1
        return out

    def aug(self, xyz_gt, transform):
        """
        Applies transformation matrix to pointcloud
        :param xyz_gt: original pointcloud with shape (3, height, width)
        :param transform: (4, 4) transformation matrix
        :return: Transformed pointcloud with shape (3, height, width)
        """
        orig_shape = xyz_gt.shape
        xyz = np.reshape(xyz_gt, [-1, 3])
        xyz = np.concatenate([xyz, np.ones([xyz.shape[0], 1])], axis=-1)

        xyz_t = (transform @ xyz.T).T

        xyz_t = xyz_t[:, :3] / xyz_t[:, 3, np.newaxis]
        xyz_t = np.reshape(xyz_t, orig_shape)
        return xyz_t

    def __getitem__(self, index):
        """
        Returns one sample for training
        :param index: index of entry
        :return: dict containing sample data
        """
        entry = self.entries[index]

        gt_transform = np.array(entry['proper_transform'])
        orig_transform = np.array(entry['orig_transform'])

        if gt_transform[0, 1] < 0.0:
            gt_transform[:, :2] *= -1

        if self.split == 'train':
            aug_transform = self.get_aug_transform()
            transform = aug_transform @ gt_transform
        else:
            transform = gt_transform

        transform = transform.astype(np.float32)

        rot = Rotation.from_matrix(transform[:3, :3])
        rotvec = torch.from_numpy(rot.as_rotvec())
        t = torch.from_numpy(transform[:3, 3])

        if self.preload:
            xyz = entry['xyz']
        else:
            xyz = self.load_xyz(entry)

        if self.split == 'train':
            xyz = self.aug(xyz, aug_transform)

        xyz = xyz.astype(np.float32)

        if self.noise_sigma is not None:
            xyz += self.noise_sigma * np.random.randn(*xyz.shape)

        return {'xyz': xyz, 'bin_rotvec': rotvec, 'bin_translation': t, 'bin_transform': torch.from_numpy(transform),
                'orig_transform': torch.from_numpy(orig_transform), 'txt_path': entry['txt_path']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', help='Path to dataset json file.')
    args = parser.parse_args()
    json_path = args.json

    dataset = Dataset(json_path, 'train', 258, 193, preload=False, noise_sigma=0.0, random_rot=True)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    for item in data_loader:
        print(item['xyz'].size())
        xyz = item['xyz'][0].cpu().detach().numpy()

        print(np.mean(xyz))

        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(xyz[0].ravel(), xyz[1].ravel(), xyz[2].ravel(), marker='o')

        #plt.show()
