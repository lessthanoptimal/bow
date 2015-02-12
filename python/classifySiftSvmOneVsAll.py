__author__ = 'pja'

from create_input_set import *
import pickle
from siftclassify import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import numpy as np

sift_conf = {'step':8,'size':4,'fast':True}
kmeans_conf = {'K':50,'num_seeds':1,'max_niters':200}
maxFeaturesPerImage = 150
svm_conf = {'dual':False,'C':0.1}

# Number of images it will include in the training and test set
maxTrainingImages = -1

# compute scaling parameters for data such that it will be 0 to 1
def computeScaleParameters( x ):
    N = len(x)
    M = len(x[0])

    params = []

    for i in xrange(M):
        low = high = x[0][i]
        for j in xrange(N):
            value = x[j][i]
            if value < low:
                low = value
            elif value > high:
                high = value

        # TODO remove hack to disable scaling
        # low = 0
        # high = 1
        params.append({'low':low,'spread':high-low})

    return params


# scale features to have a range of 0 to 1
def scaleFeatureData( x , params ):
    M = len(params)

    for i in xrange(M):
        low = params[i]['low']
        spread = params[i]['spread']

        for j in xrange(len(x)):
            x[j][i] = (x[j][i] - low)/spread

def convertToSvmFormatFeature( histograms ):
    N = len(histograms)
    x = []

    for i in xrange(N):
        x.append(histograms[i]['histogram'][:])

    return x

def convertAndScaleToSvmFormat( scaleParam, histogram ):
    x = [0]*len(histogram)
    for i in xrange(len(histogram)):
        spread = scaleParam[i]['spread']
        low = scaleParam[i]['low']
        x[i] = (histogram[i]-low)/spread
    return x

def trainOneVsAll( histograms ):

    xAll = convertToSvmFormatFeature(histograms)
    scaleParam = computeScaleParameters(xAll)
    scaleFeatureData(xAll,scaleParam)

    yAll = [ x['label'] for x in histograms ]

    svm = OneVsRestClassifier(LinearSVC(dual=svm_conf['dual'],C=svm_conf['C']))
    svm.fit(xAll,yAll)

    out = {'scaleParam':scaleParam,'svm':svm}
    return out

def predictOneVsAll( param , histogram ):
    x = convertAndScaleToSvmFormat(param['scaleParam'],histogram)

    svm = param['svm']
    return svm.predict(x)

class BowSiftLinearSVM:
    def __init__(self, svmParam , verbose=False):
        self.svmParam = svmParam
        self.verbose = verbose

    def train(self, labeled ):
        dictionaryName = 'dictionary{0:d}.p'.format(kmeans_conf['K'])
        histogramsName = 'histograms{0:d}.p'.format(kmeans_conf['K'])

        self.numLabels = labeled['numLabels']

        try:
            self.vocabulary = pickle.load(open( dictionaryName, "rb" ))
            print 'Loaded vocabulary! file name = '+dictionaryName
        except IOError:
            print 'Computing words'
            allWords = computeAllWords(labeled['paths'],maxFeaturesPerImage,sift_conf,self.verbose)
            print 'Creating the vocabulary'
            self.vocabulary = create_vocabulary(allWords,kmeans_conf,self.verbose)
            pickle.dump(self.vocabulary,open( dictionaryName, "wb" ))
        except:
            raise

        try:
            self.histograms = pickle.load(open( histogramsName, "rb" ))
            print 'Loaded NN graph! file name = '+histogramsName
        except IOError:
            print 'Computing histograms and NN graph from training set'
            self.histograms = self.trainHistograms(labeled['paths'])
            pickle.dump(self.histograms,open( histogramsName, "wb" ))
        except:
            raise

        self.svm = trainOneVsAll(self.histograms)

    def trainHistograms( self, paths ):
        histograms = []
        for p in paths:
            filePath = p['path']
            label = p['label']
            if self.verbose:
                print 'training histogram for '+filePath
            h = compute_histogram(filePath,self.vocabulary,kmeans_conf,sift_conf)
            histograms.append({'histogram':h,'label':label})

        return histograms

    def classify(self, filePath):

        target = compute_histogram(filePath,self.vocabulary,kmeans_conf,sift_conf)

        return predictOneVsAll(self.svm,target)

# score the test set
labeledTraining = truncateFileList(findLabeledImages("../brown/data/train"),maxTrainingImages)
labeledTest = truncateFileList(findLabeledImages("../brown/data/test"),maxTrainingImages)

# Linear one-vs-all  {'K':50,'num_seeds':1,'max_niters':200}
# '-s 0 -t 2 -c 2 -g 2' {'dual':True,'C':0.01}        0.523
# '-s 0 -t 2 -c 2 -g 2' {'dual':True,'C':0.1}         0.589
# '-s 0 -t 2 -c 2 -g 2' {'dual':True,'C':1}           0.614
# '-s 0 -t 2 -c 2 -g 2' {'dual':False,'C':10}         0.612

# Linear one-vs-all  {'K':400,'num_seeds':1,'max_niters':200}
# '-s 0 -t 2 -c 2 -g 2' {'dual':False,'C':0.01}       0.666
# '-s 0 -t 2 -c 2 -g 2' {'dual':False,'C':0.1}        0.705
# '-s 0 -t 2 -c 2 -g 2' {'dual':False,'C':10}         0.623
# '-s 0 -t 2 -c 2 -g 2' {'dual':False,'C':100}        0.615

evaluateClassifier(BowSiftLinearSVM('-s 0 -t 2 -c 2 -g 2',True),labeledTraining,labeledTest)

print 'Done!'
