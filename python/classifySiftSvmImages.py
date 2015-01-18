__author__ = 'pja'

from create_input_set import *
import pickle
from siftclassify import *
from svmutil import *
from svm import *

import numpy as np

sift_conf = {'step':8,'size':4,'fast':True}
kmeans_conf = {'K':50,'num_seeds':1,'max_niters':200}
maxFeaturesPerImage = 150

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

def convertToSvmFormatType( target , histograms ):
    N = len(histograms)
    y = [0]*N  # will be assigned values of 1 if target or -1 if not

    for i in xrange(N):
        if histograms[i]['label'] == target:
            y[i] = 1
        else:
            y[i] = -1

    return y

def extractTrainOneVsOne( setA , setB , xAll , histograms):
    y = []
    x = []

    for i in xrange( len(xAll)):
        label = histograms[i]['label']
        if label == setA:
            y.append(1)
        elif label == setB:
            y.append(-1)
        else:
            continue
        x.append( xAll[i] )
    return y,x

def trainOneVsOne( numLabels , histograms, svmParam ):

    xAll = convertToSvmFormatFeature(histograms)
    scaleParam = computeScaleParameters(xAll)
    scaleFeatureData(xAll,scaleParam)

    svm = []
    for i in xrange(numLabels):
        svmI = []
        for j in xrange(i,numLabels):
            print 'training '+str(i)+" , "+str(j)
            y,x = extractTrainOneVsOne(i,j,xAll,histograms)
            m = svm_train(y, x, svmParam)
            svmI.append(m)
        svm.append(svmI)

    out = {'scaleParam':scaleParam,'svm':svm}
    return out

def predictOneVsOne( param , histogram ):
    x = convertAndScaleToSvmFormat(param['scaleParam'],histogram)
    # Convert a Python-format instance to svm_nodearray, a ctypes structure
    x0, max_idx = gen_svm_nodearray(x)

    svm = param['svm']
    N = len(svm)
    histogram = [0]*N

    for i in xrange(N):
        for j in xrange(i,N):
            # print "predict {:d} {:d}".format(i,j)
            m = svm[i][j-i]
            label = libsvm.svm_predict(m, x0)
            if label == 1:
                histogram[i] += 1
            elif label == -1:
                histogram[j] += 1
            else:
                print "shit bug"

    return np.argmax(histogram)



def trainOneVsAll( numLabels , histograms, svmParam ):

    x = convertToSvmFormatFeature(histograms)
    scaleParam = computeScaleParameters(x)
    scaleFeatureData(x,scaleParam)

    svm = []
    for i in xrange(numLabels):
        print 'i = '+str(i)
        y = convertToSvmFormatType(i,histograms)
        m = svm_train(y, x, svmParam)
        svm.append(m)

    out = {'scaleParam':scaleParam,'svm':svm}
    return out

def predictOneVsAll( param , histogram ):
    # scale the input histogram
    x = convertAndScaleToSvmFormat(param['scaleparam'],histogram)

    N = len(param['svm'])
    for i in xrange(N):
        m = param['svm'][i]

        # Convert a Python-format instance to svm_nodearray, a ctypes structure
        x0, max_idx = gen_svm_nodearray(x)
        label = libsvm.svm_predict(m, x0)
        if label == 1: # TODO change this.  It should be based off of some sort of error metric!
            return i

    return -1 # let it know it has no idea



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

        # compute parameters for an 1-vs-1 SVM classifier
        self.svm = trainOneVsOne(self.numLabels,self.histograms,self.svmParam)

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

        return predictOneVsOne(self.svm,target)

# score the test set
labeledTraining = truncateFileList(findLabeledImages("../brown/data/train"),maxTrainingImages)
labeledTest = truncateFileList(findLabeledImages("../brown/data/test"),maxTrainingImages)

# kmeans_conf = {'K':50,'num_seeds':1,'max_niters':200}
# '-s 0 -t 2 -c 200 -g 1'    scaling 0.622
# '-s 0 -t 2 -c 200 -g 1' no scaling 0.630
# '-s 0 -t 2 -c 200 -g 2'    scaling 0.629
# '-s 0 -t 2 -c 2 -g 2'      scaling 0.636
# '-s 0 -t 2 -c 2000 -g 2'   scaling 0.629
# '-s 0 -t 2 -c 0.1 -g 2'    scaling 0.538

# kmeans_conf = {'K':400,'num_seeds':1,'max_niters':200}
# '-s 0 -t 2 -c 2 -g 2'      scaling 0.364
# '-s 0 -t 2 -c 1e5 -g 2'    scaling 0.364

evaluateClassifier(BowSiftLinearSVM('-s 0 -t 2 -c 1e5 -g 2',True),labeledTraining,labeledTest)

# TODO
# TODO SVM improve training to avoid over fitting

print 'Done!'
