__author__ = 'pja'

from os import listdir
from os.path import *
import random
import matplotlib.pyplot as plt
import numpy as np


def findLabeledImages( base ):
    dirs = [ f for f in listdir(base) if not isfile(join(base,f))]

    ret = []

    dirs = sorted(dirs, key=str.lower)

    for idx,d in enumerate(dirs):
        imgDir = join(base,d)
        ret = ret + [{'label':idx , 'path':join(imgDir,f)} for f in listdir(imgDir) if isfile(join(imgDir,f))]

    return {'numLabels':len(dirs),'paths':ret}

def truncateFileList( labeled , N=-1 ):
    # remove all but the first N images
    if -1 < N < len(labeled['paths']):
        random.shuffle(labeled['paths'])
        labeled['paths'] = labeled['paths'][0:N]

    return labeled

def evaluateClassifier( classifier , labeledTraining , labeledTest ):

    classifier.train(labeledTraining)

    numLabels = labeledTraining['numLabels']

    confusion = np.zeros([numLabels,numLabels]) # declare confusion matrix
    labelCount = [0]*numLabels

    # confusion = np.random.random([numLabels,numLabels])

    total = 0
    totalCorrect = 0
    totalUnknown = 0
    for p in labeledTest['paths']:
        filePath = p['path']
        label = p['label']
        found = classifier.classify(filePath)
        labelCount[label] += 1
        if found >= 0:
            confusion[label,found] += 1
        else:
            totalUnknown += 1

        total += 1

        if found == label:
            totalCorrect += 1

        print '  evaluated {:5d} / {:5d}  correct {:.3f}  unknown {:.3f}  actual label {:d}'.\
            format(total, len(labeledTest['paths']), totalCorrect/float(total),totalUnknown/float(total),label)

    for i in xrange(numLabels):
        for j in xrange(numLabels):
            if labelCount[i] > 0:
                confusion[i, j] /= labelCount[i]

    # plot the confusion matrix
    strLabels = [str(x) for x in xrange(numLabels)]
    fig, ax = plt.subplots()

    ax.axis([0,numLabels,0,numLabels])
    ax.pcolor(confusion, cmap='RdBu', vmin=0, vmax=1)

    # center the tick labels
    ax.set_xticks(np.arange(numLabels)+0.5, minor=False)
    ax.set_yticks(np.arange(numLabels)+0.5, minor=False)
    ax.set_xticklabels(strLabels, minor=False)
    ax.set_yticklabels(strLabels, minor=False)
    ax.set_aspect('equal')
    plt.show()




