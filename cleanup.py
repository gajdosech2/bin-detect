import glob
import os

if __name__ == '__main__':
    """
    Cleans all generated prediction files.
    """
    entries = glob.iglob('datasets/**/prediction_*', recursive=True)
    for entry in entries:
        os.remove(entry)
