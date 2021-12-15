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
    numerator = np.clip(numerator, -2, 2)
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


def read_icp_file(file):
    with open(file, 'r') as tfile:
        P = tfile.read().strip().replace('\n', ', ').split(', ')
        R = np.array([[float(P[0]), float(P[4]), float(P[8])],
                      [float(P[1]), float(P[5]), float(P[9])],
                      [float(P[2]), float(P[6]), float(P[10])]])
        t = np.array([float(P[12]), float(P[13]), float(P[14])])
        return R, t


def write_refined_file(path, number, R44):
    with open(path + '/refined_scan_' + number + '.txt', 'w') as rf:
        print(f'{R44[0][0]} {R44[1][0]} {R44[2][0]} 0.0 '
              f'{R44[0][1]} {R44[1][1]} {R44[2][1]} 0.0 '
              f'{R44[0][2]} {R44[1][2]} {R44[2][2]} 0.0 '
              f'{R44[0][3]} {R44[1][3]} {R44[2][3]} 1.0', file=rf)


def evaluate(args_path):
    gt_files = glob.iglob(args_path + '/**/*.txt', recursive=True)
    good_gt_files = [f for f in gt_files if not any(sub in f for sub in ['bad', 'catas', 'ish', 'pred', 'icp', 'refined']) and any(sub in f for sub in ['scan_', 'gt_'])]
    eTE_list = []
    eTE_list_icp = []
    eRE_list = []
    eRE_list_icp = []
    eGD_list = []
    eGD_list_icp = []

    counter_better = 0
    counter_worse = 0
    for file in good_gt_files:
        path, gt_file = os.path.split(file)
        index = gt_file.rfind("_")
        number = gt_file[index+1:-4]
        
        pr_file = number + '.txt'
        if not os.path.isfile(path + '/' + pr_file):
            if os.path.isfile(path + '/' + 'prediction_scan_' + number + '.txt'):
                pr_file = 'prediction_scan_' + number + '.txt'
            else:
                print("Prediction file not found for " + file)
                continue

        pr_R, pr_t = read_transform_file(path + '/' + pr_file)

        gt_R1, gt_t = read_transform_file(path + '/' + gt_file)
        if np.linalg.det(gt_R1) < 0:
            gt_R1[:, 1] *= -1
        gt_R2 = np.matrix.copy(gt_R1)
        gt_R2[:, :2] *= -1

        eTE_list.append(calculate_eTE(gt_t, pr_t))
        if np.array_equal(pr_R, np.eye(3, 3)):
            eRE_list.append(float("inf"))
        else:
            eRE_list.append(min(calculate_eRE(gt_R1, pr_R), calculate_eRE(gt_R2, pr_R)))
        eGD_list.append(min(calculate_eGD(gt_R1, pr_R), calculate_eGD(gt_R2, pr_R)))

        pr_RICP = np.copy(pr_R)
        pr_tICP = np.copy(pr_t)
        icp_file = 'icp_scan_' + number + '.txt'
        if os.path.isfile(path + '/' + icp_file):
            pr_RICP, pr_tICP = read_icp_file(path + '/' + icp_file)

            pr_44R = [[pr_R[0][0], pr_R[0][1], pr_R[0][2], pr_t[0]],
                      [pr_R[1][0], pr_R[1][1], pr_R[1][2], pr_t[1]],
                      [pr_R[2][0], pr_R[2][1], pr_R[2][2], pr_t[2]],
                      [0.0, 0.0, 0.0, 1.0]]
            pr_44I = [[pr_RICP[0][0], pr_RICP[0][1], pr_RICP[0][2], pr_tICP[0]],
                      [pr_RICP[1][0], pr_RICP[1][1], pr_RICP[1][2], pr_tICP[1]],
                      [pr_RICP[2][0], pr_RICP[2][1], pr_RICP[2][2], pr_tICP[2]],
                      [0.0, 0.0, 0.0, 1.0]]

            pr_44F = np.matmul(pr_44I, pr_44R)
            pr_RICP = pr_44F[:3, :3]
            pr_tICP = pr_44F[:3, 3]
            write_refined_file(path, number, pr_44F)

        if min(calculate_eRE(gt_R1, pr_R), calculate_eRE(gt_R2, pr_R)) > min(calculate_eRE(gt_R1, pr_RICP), calculate_eRE(gt_R2, pr_RICP)) and \
            calculate_eTE(gt_t, pr_t) > calculate_eTE(gt_t, pr_tICP):
            print('BETTER: ' + path + '/' + gt_file)
            counter_better += 1
        else:
            print('WORSE: ' + path + '/' + gt_file)
            counter_worse += 1

        eRE_list_icp.append(min(calculate_eRE(gt_R1, pr_RICP), calculate_eRE(gt_R2, pr_RICP)))
        eTE_list_icp.append(calculate_eTE(gt_t, pr_tICP))
        eGD_list_icp.append(min(calculate_eGD(gt_R1, pr_RICP), calculate_eGD(gt_R2, pr_RICP)))

    print(len(eTE_list))
    print(counter_better)
    print(counter_worse)

    print(f'MEAN eTE {mean(eTE_list)}, eRE: {mean(eRE_list)}, eGD: {mean(eGD_list)}')
    print(f'STD eTE {np.std(eTE_list)}, eRE: {np.std(eRE_list)}, eGD: {np.std(eGD_list)}')
    print(f'MEDIAN eTE {median(eTE_list)}, eRE: {median(eRE_list)}, eGD: {median(eGD_list)}')
    print(f'MIN eTE {min(eTE_list)}, eRE: {min(eRE_list)}, eGD: {min(eGD_list)}')
    print(f'MAX eTE {max(eTE_list)}, eRE: {max(eRE_list)}, eGD: {max(eGD_list)}')

    print('AFTER ICP')
    print(f'MEAN eTE {mean(eTE_list_icp)}, eRE: {mean(eRE_list_icp)}, eGD: {mean(eGD_list_icp)}')
    print(f'STD eTE {np.std(eTE_list_icp)}, eRE: {np.std(eRE_list_icp)}, eGD: {np.std(eGD_list_icp)}')
    print(f'MEDIAN eTE {median(eTE_list_icp)}, eRE: {median(eRE_list_icp)}, eGD: {median(eGD_list_icp)}')
    print(f'MIN eTE {min(eTE_list_icp)}, eRE: {min(eRE_list_icp)}, eGD: {min(eGD_list_icp)}')
    print(f'MAX eTE {max(eTE_list_icp)}, eRE: {max(eRE_list_icp)}, eGD: {max(eGD_list_icp)}')

    return eTE_list, eRE_list, eGD_list, eTE_list_icp, eRE_list_icp, eGD_list_icp
       

if __name__ == '__main__':
    """
    Example usage: python evaluate.py path/to/dataset_with_predictions
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to dataset root folder.')
    evaluate(parser.parse_args().path)
