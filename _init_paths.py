"""Set up paths."""

import os.path as osp
import os
import sys
from ctypes import *
import matlab.engine

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add pycaffe
caffe_path = osp.join(this_dir, 'build', 'pycaffe')
add_path(caffe_path)

# Add search dll
dll_path = osp.join(this_dir, 'build', 'search')
add_path(dll_path)

# Load dynamic library compiled from "search (c++)" and use as search similar images
os.chdir(dll_path)
search_dll = windll.LoadLibrary('Search.dll')
search = search_dll[1]
os.chdir(this_dir)

# Add matlab script
mat_path = osp.join(this_dir, 'source', 'pca')
eng = matlab.engine.start_matlab()
eng.addpath(mat_path)

def get_search_func():
    return search

def get_pca_func():
    return eng.pca_func

