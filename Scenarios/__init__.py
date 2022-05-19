'''
Functions for reading data according to the given scenario name

Using:
os: Built-in package of Python
Python: 3.9
'''
from imp import load_source
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return load_source('', pathname)