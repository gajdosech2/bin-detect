import argparse
import glob
import os
import numpy as np
from statistics import mean, median
from scipy.linalg import logm


def calculate_eTE(gt_t, pr_t):
    return np.linalg.norm((pr_t-gt_t), ord=2)/10
   
   
def calculate_eRE(gt_R, pr_R):
    numerator = np.trace(np.matmul(gt_R, np.linalg.inv(pr_R))) - 1
    return np.arccos(numerator/2)
  
  
def calculate_eGD(gt_R, pr_R):
    argument = logm(np.matmul(gt_R, np.transpose(pr_R)))
    numerator = np.linalg.norm(argument, ord='fro')
    return numerator/(2**.5)


def read_transform_file(file):
    with open(file, 'r') as tfile:
        P = tfile.readline().strip().split(' ')
        R = np.array([[float(P[0]), float(P[4]), float(P[8])],
                      [float(P[1]), float(P[5]), float(P[9])],
                      [float(P[2]), float(P[6]), float(P[10])]])
        t = np.array([float(P[12]), float(P[13]), float(P[14])])
    return R, t


def evaluate(args):
    gt_files = glob.iglob(args.path + '/**/*.txt', recursive=True)
    good_gt_files = [f for f in gt_files if not any(sub in f for sub in ['bad', 'catas', 'ish', 'pred']) and any(sub in f for sub in ['scan_', 'gt_'])]
    eTE_list = []
    eRE_list = []
    eGD_list = []

    for file in good_gt_files:
        path, gt_file = os.path.split(file)
        index = gt_file.rfind("_")
        number = gt_file[index+1:-4]
        
        pr_file = number + '.txt'
        
        if not os.path.isfile(path + '/' + pr_file):
            if os.path.isfile(path + '/' + 'prediction_scan_' + number + '.txt'):
                pr_file = 'prediction_scan_' + number + '.txt'
            else:
                continue

        gt_R1, gt_t = read_transform_file(path + '/' + gt_file)
        pr_R, pr_t = read_transform_file(path + '/' + pr_file)
        
        gt_R2 = np.matrix.copy(gt_R1)
        gt_R2[:, :2] *= -1

        eTE_list.append(calculate_eTE(gt_t, pr_t))
        eRE_list.append(min(calculate_eRE(gt_R1, pr_R), calculate_eRE(gt_R2, pr_R)))
        eGD_list.append(min(calculate_eGD(gt_R1, pr_R), calculate_eGD(gt_R2, pr_R)))
        
    print(len(eTE_list))
    print(f'MEAN eTE {mean(eTE_list)}, eRE: {mean(eRE_list)}, eGD: {mean(eGD_list)}')
    print(f'STD eTE {np.std(eTE_list)}, eRE: {np.std(eRE_list)}, eGD: {np.std(eGD_list)}')
    print(f'MEDIAN eTE {median(eTE_list)}, eRE: {median(eRE_list)}, eGD: {median(eGD_list)}')
    print(f'MIN eTE {min(eTE_list)}, eRE: {min(eRE_list)}, eGD: {min(eGD_list)}')
    print(f'MAX eTE {max(eTE_list)}, eRE: {max(eRE_list)}, eGD: {max(eGD_list)}')
       

if __name__ == '__main__':
    """
    Example usage: python evaluate.py path/to/dataset_with_predictions
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to dataset root folder.')
    evaluate(parser.parse_args())
