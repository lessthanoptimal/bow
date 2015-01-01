__author__ = 'pja'

from os import listdir
from os.path import *
from scipy import misc
from math import *
import numpy as np


def findLabeledImages( base ):
    dirs = [ f for f in listdir(base) if not isfile(join(base,f))]

    ret = []

    dirs = sorted(dirs, key=str.lower)

    for idx,d in enumerate(dirs):
        imgDir = join(base,d)
        ret = ret + [{'label':idx , 'path':join(imgDir,f)} for f in listdir(imgDir) if isfile(join(imgDir,f))]

    return {'numLabels':len(dirs),'paths':ret}







