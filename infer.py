import os

import torch
import cv2
import numpy as np
from dataset import Dataset

from network import Network, parse_command_line, load_model
from scipy.spatial.transform.rotation import Rotation
from torch.utils.data import DataLoader
from shutil import copyfile


def infer(args, export_to_folder=False):
    model = load_model(args).eval()

    dir_path = os.path.dirname(args.path)

    val_dataset = Dataset(args.path, 'val', args.input_width, args.input_height, preload=not args.no_preload)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    np.set_printoptions(suppress=True)

    with torch.no_grad():
        for sample in val_loader:
            pred_zs, pred_ys, pred_ts = model(sample['xyz'].cuda())
            gt_transforms = sample['orig_transform']

            for i in range(len(pred_zs)):
                print(20  * '*')
                print("GT:")
                gt_transform = gt_transforms[i].cpu().numpy()
                print("Det: ", np.linalg.det(gt_transform))
                print(gt_transform)

                z = pred_zs[i].cpu().numpy()
                z /= np.linalg.norm(z)

                y = pred_ys[i].cpu().numpy()
                y = y - np.dot(z, y)*z
                y /= np.linalg.norm(y)

                x = np.cross(y, z)

                transform = np.zeros([4, 4])
                transform[:3, 0] = x
                transform[:3, 1] = y
                transform[:3, 2] = z

                transform[:3, 3] = pred_ts[i].cpu().numpy()
                transform[3, 3] = 1
                print("Predict:")
                print("Det: ", np.linalg.det(transform))
                print(transform)

                txt_path = sample['txt_path'][i]
                txt_name = 'prediction_{}'.format(os.path.basename(txt_path))
                txt_dir = os.path.dirname(txt_path)
                save_txt_path = os.path.join(dir_path, txt_dir, txt_name)
                np.savetxt(save_txt_path, transform.T.ravel(), fmt='%1.6f', newline=' ')

                if export_to_folder:
                    """
                    Copies .txt predictions, .cogs scans and bin .stl into standalone folder with Inference suffix
                    """
                    export_path = dir_path + 'Inference'
                    if not os.path.isdir(export_path):
                        os.mkdir(export_path)

                    if not os.path.isdir(os.path.join(export_path, txt_dir)):
                        os.mkdir(os.path.join(export_path, txt_dir))

                    export_txt_path = os.path.join(export_path, txt_dir, txt_name)
                    np.savetxt(export_txt_path, transform.T.ravel(), fmt='%1.6f', newline=' ')
                    scan_name = txt_name[11:-4] + '.cogs'
                    copyfile(os.path.join(dir_path, txt_dir, scan_name), os.path.join(export_path, txt_dir, scan_name))

                    if not os.path.exists(os.path.join(export_path, txt_dir, 'bin.stl')):
                        copyfile(os.path.join(dir_path, txt_dir, 'bin.stl'), os.path.join(export_path, txt_dir, 'bin.stl'))


if __name__ == '__main__':
    """
    Runs inference and writes prediction txt files.
    Example usage: python infer.py --no_preload -r 200 -iw 258 -ih 193 -b 32 /path/to/MLBinsDataset/EXR/dataset.json
    """
    args = parse_command_line()
    infer(args)
