__author__ = 'pja'

from create_input_set import findLabeledImages
from sklearn.neighbors import NearestNeighbors
from scipy import misc
from math import *
import numpy as np


def getTinyImages( paths , width ):
    desc = []
    for p in paths:
        label = p['label']
        filePath = p['path']
        image = misc.imread(filePath)
        d = convertImage(misc.imresize(image,[width,width],interp='bicubic'))

        # make it zero mean
        mean = np.mean(d)
        d = [x-mean for x in d]
        # make it unit length
        div = sqrt(sum([x*x for x in d]))
        d = [x/div for x in d]

        desc.append( {'label':label, 'desc':d} )

    return desc

def convertImage( image ):
    height = image.shape[0]
    width = image.shape[1]

    d = [0.0]*(width*height)
    for i in range(width):
        for j in range(width):
            d[i*width+j] = image[i,j]
    return d

def trainNearestNeighbor( imageDescs , numLabels, numNeighbors ):
    labels = [image['label'] for image in imageDescs]
    x = []
    for image in imageDescs:
        x.append(image['desc'])
    x = np.array(x)
    # nbrs = NearestNeighbors(n_neighbors=numNeighbors, algorithm='ball_tree').fit(x)
    nbrs = NearestNeighbors(n_neighbors=numNeighbors, algorithm='kd_tree').fit(x)


    return {'nbrs':nbrs,'labels':labels,'numLabels':numLabels}

def classifyNearestNeighbor( desc , classifierData ):
    nbrs = classifierData['nbrs']
    labels = classifierData['labels']
    numLabels = classifierData['numLabels']

    distance,indices = nbrs.kneighbors(desc['desc'])

    hits = [0]*numLabels
    for idx,i in enumerate(indices.flat):
        # hits[labels[i]] = hits[labels[i]] + 1 # uniform weighting
        hits[labels[i]] = hits[labels[i]] + 1.0/(distance[0,idx]+0.2) # weight by distance

    return np.argmax(hits)

#  1 = 21.4
#  3 = 20.?%   22.51% weighted
#  6 = 21.5%   22.47% weighted
#  8 = 21.1%   22.98% weighted
# 10 = 21.2%   22.74% weighted
# 20 = 20.?%   20.97% weighted
# 40 = 19.?%

print 'Finding Training Images'
labeledTraining = findLabeledImages("../brown/data/train")
print 'Computing Tiny Descriptors for Training Set'
labeledTrainingDescs = getTinyImages(labeledTraining['paths'],16)
print 'Training on '+str(len(labeledTrainingDescs))
classifierData = trainNearestNeighbor(labeledTrainingDescs,labeledTraining['numLabels'],8)
print 'Finding Test Images'
labeledTest = findLabeledImages("../brown/data/test")
print 'Computing Tiny Descriptors for Test Set'
labeledTestDescs = getTinyImages(labeledTest['paths'],16)
print 'Classifying Test Data:  total '+str(len(labeledTestDescs))
foundLabels = [classifyNearestNeighbor(x,classifierData) for x in labeledTestDescs]
print 'Computing Statistics'
numCorrect = 0
for i in range(len(foundLabels)):
    expected = labeledTestDescs[i]['label']
    found = foundLabels[i]
    if expected == found:
        numCorrect = numCorrect + 1

percent = 100.0*numCorrect/len(foundLabels)

print 'Percent Correct Label '+str(percent)