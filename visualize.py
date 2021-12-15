import argparse
import glob
import os


def visualize(args, prefix='prediction_'):
    txt_files = glob.iglob(args.path + '/**/*.txt', recursive=True)
    good_txt_files = [f for f in txt_files if
                      'bad' not in f and 'catas' not in f and 'ish' not in f]
    if not os.path.isfile('Wizard.exe'):
        print('Wizard.exe not found.')
        return
    for file in good_txt_files:
        path, txt_file = os.path.split(file)
        trans_file = txt_file if args.ground_truth else prefix + txt_file
        if not os.path.isfile(path + '/' + trans_file):
            continue
        cogs_file = txt_file[:-4] + '.cogs'
        print('Visualizing: ', cogs_file)
        os.system('Wizard.exe ' +
                  path + '/bin.stl ' +
                  path + '/' + cogs_file + ' ' +
                  path + '/' + trans_file)


if __name__ == '__main__':
    """
    Example usage: python visualize.py path/to/dataset_with_predictions
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to dataset root folder.')
    parser.add_argument('-gt', '--ground_truth', action='store_true', help='whether to visualize GT instead of prediction', default=False)
    visualize(parser.parse_args())
