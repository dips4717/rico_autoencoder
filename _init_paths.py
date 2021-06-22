# -*- coding: utf-8 -*-


import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

data_dir = '/mnt/scratch/Dipu/RICO/semantic_annotations'

currentPath = os.path.dirname(os.path.realpath(__file__))

# Add lib to PYTHONPATH
libPath = os.path.join(currentPath, 'lib')
add_path(libPath)