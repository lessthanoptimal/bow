__author__ = 'pja'

from create_input_set import *
import pickle
from siftclassify import *
from svmutil import *
from svm import *

from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

import numpy as np

sift_conf = {'step':8,'size':4,'fast':True}
kmeans_conf = {'K':400,'num_seeds':1,'max_niters':200}
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

def extractTrainOneVsOne( setA , setB , xAll , histograms):
    y = []
    x = []

    for i in xrange( len(xAll)):
        label = histograms[i]['label']
        if label == setA:
            y.append(1)
        elif label == setB:
            y.append(0)
        else:
            continue
        x.append( xAll[i] )
    return y,x

def trainOneVsOne2( histograms ):

    xAll = convertToSvmFormatFeature(histograms)
    scaleParam = computeScaleParameters(xAll)
    scaleFeatureData(xAll,scaleParam)

    xAll = np.array(xAll)

    yAll = [ x['label'] for x in histograms ]
    yAll = np.array(yAll)

    # svm = OneVsOneClassifier(LinearSVC(random_state=0,dual=svm_conf['dual'],C=svm_conf['C']))
    gammaBase = 1.0/kmeans_conf['K']
    # svm = OneVsOneClassifier(sklearn.svm.SVC(C=100, gamma=10*gammaBase,kernel='rbf'))
    svm = OneVsOneClassifier(sklearn.svm.SVC(C=1000, gamma=gammaBase,kernel='sigmoid'))
    svm.fit(xAll,yAll)

    out = {'scaleParam':scaleParam,'svm':svm}
    return out

def predictOneVsOne2( param , histogram ):
    x = convertAndScaleToSvmFormat(param['scaleParam'],histogram)
    x = np.array([x])

    svm = param['svm']
    return svm.predict(x)[0]

def trainOneVsOne( numLabels , histograms):

    xAll = convertToSvmFormatFeature(histograms)
    scaleParam = computeScaleParameters(xAll)
    scaleFeatureData(xAll,scaleParam)

    svm = []
    for i in xrange(numLabels):
        svmI = []
        for j in xrange(i+1,numLabels):
            print 'training '+str(i)+" , "+str(j)
            y,x = extractTrainOneVsOne(i,j,xAll,histograms)
            m = LinearSVC(random_state=0,dual=svm_conf['dual'],C=svm_conf['C'])
            m.fit(x,y)
            svmI.append(m)
        svm.append(svmI)

    out = {'scaleParam':scaleParam,'svm':svm}
    return out

def predictOneVsOne( param , histogram ):
    x = convertAndScaleToSvmFormat(param['scaleParam'],histogram)

    svm = param['svm']
    N = len(svm)
    histogram = [0]*N

    for i in xrange(N):
        for j in xrange(i+1,N):
            # print "predict {:d} {:d}".format(i,j)
            m = svm[i][j-i-1]
            label = m.predict(x)
            if label == 1:
                histogram[i] += 1
            elif label == 0:
                histogram[j] += 1
            else:
                print "shit bug"

    return np.argmax(histogram)

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
        # self.svm = trainOneVsOne(self.numLabels,self.histograms)
        self.svm = trainOneVsOne2(self.histograms)

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

        # return predictOneVsOne(self.svm,target)
        return predictOneVsOne2(self.svm,target)

# score the test set
labeledTraining = truncateFileList(findLabeledImages("../brown/data/train"),maxTrainingImages)
labeledTest = truncateFileList(findLabeledImages("../brown/data/test"),maxTrainingImages)

# SKLearn One vs One Linear
# {'K':400,'num_seeds':1,'max_niters':200}  {'dual':False,'C':0.1}     0.718

# SKLearn One vs One RBF
# {'K':400,'num_seeds':1,'max_niters':200}  C=1                        0.596
# {'K':400,'num_seeds':1,'max_niters':200}  C=10                       0.697
# {'K':400,'num_seeds':1,'max_niters':200}  C=100                      0.715
# {'K':400,'num_seeds':1,'max_niters':200}  C=1000                     0.701

# {'K':400,'num_seeds':1,'max_niters':200}  C=100   sigma=0.01*base    0.594
# {'K':400,'num_seeds':1,'max_niters':200}  C=100   sigma=0.1*base     0.697
# {'K':400,'num_seeds':1,'max_niters':200}  C=100   sigma=1*base       0.715
# {'K':400,'num_seeds':1,'max_niters':200}  C=100   sigma=10*base      0.715
# {'K':400,'num_seeds':1,'max_niters':200}  C=100   sigma=100*base     0.742
# {'K':400,'num_seeds':1,'max_niters':200}  C=100   sigma=1000*base    0.294

# SKLearn One vs One Sigmoid
# {'K':400,'num_seeds':1,'max_niters':200}  C=1    sigma=1*base        0.593
# {'K':400,'num_seeds':1,'max_niters':200}  C=10   sigma=1*base        0.677
# {'K':400,'num_seeds':1,'max_niters':200}  C=100  sigma=1*base        0.712
# {'K':400,'num_seeds':1,'max_niters':200}  C=1000 sigma=1*base        0.697

# Mine One vs One Linear
# {'K':400,'num_seeds':1,'max_niters':200}  {'dual':False,'C':0.1}     0.720

evaluateClassifier(BowSiftLinearSVM('-s 0 -t 2 -c 2 -g 2',True),labeledTraining,labeledTest)


print 'Done!'
