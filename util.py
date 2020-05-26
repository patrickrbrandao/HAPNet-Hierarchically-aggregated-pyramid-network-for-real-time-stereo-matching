import re
import os
import sys
import glob
import logging
import datetime
import numpy as np

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def ft3d_filenames(path):
    ft3d_path = path
    ft3d_samples_filenames = {}

    with open(os.path.join(path,"train_images.txt")) as f:
        left_images_filenames = f.read().splitlines()
    right_images_filenames=[path.replace("left","right") for path in left_images_filenames]

    with open(os.path.join(path,"train_labels.txt")) as f:
        disparity_filenames = f.read().splitlines()

    ft3d_samples_filenames["TRAIN"] = [(left_images_filenames[i],
                                       right_images_filenames[i],
                                       disparity_filenames[i]) for i in range(len(left_images_filenames))]

    with open(os.path.join(path,"test_images.txt")) as f:
        left_images_filenames = f.read().splitlines()
    right_images_filenames=[path.replace("left","right") for path in left_images_filenames]

    with open(os.path.join(path,"test_labels.txt")) as f:
        disparity_filenames = f.read().splitlines()

    ft3d_samples_filenames["TEST"] = [(left_images_filenames[i],
                                       right_images_filenames[i],
                                       disparity_filenames[i]) for i in range(len(left_images_filenames))]

    return ft3d_samples_filenames

def init_logger(log_path, name="dispnet"):
    root = logging.getLogger()
    root.setLevel(logging.NOTSET)
    logfile = os.path.join(log_path, "%s-%s.log" % (name, datetime.datetime.today()))
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setLevel(logging.INFO)
    root.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.terminator = ""
    root.addHandler(consoleHandler)
    logging.debug("Logging to %s" % logfile)
