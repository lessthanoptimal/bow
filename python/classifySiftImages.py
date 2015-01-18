__author__ = 'pja'

from create_input_set import *
from sklearn.neighbors import NearestNeighbors
import pickle
from siftclassify import *

import numpy as np

sift_conf = {'step':8,'size':4,'fast':True}
kmeans_conf = {'K':50,'num_seeds':1,'max_niters':200}
maxFeaturesPerImage = 150

# Number of images it will include in the training and test set
maxTrainingImages = -1
# number of neighbors in NN classifier
numNeighbors = 10

class BowSiftKD:

    def train(self, labeled ):
        dictionaryName = 'dictionary{0:d}.p'.format(kmeans_conf['K'])
        nn_name = 'nndata{0:d}_NN{1:d}.p'.format(kmeans_conf['K'],numNeighbors)

        try:
            self.vocabulary = pickle.load(open( dictionaryName, "rb" ))
            print 'Loaded vocabulary! file name = '+dictionaryName
        except IOError:
            print 'Computing words'
            allWords = computeAllWords(labeled['paths'],maxFeaturesPerImage,sift_conf,True)
            print 'Creating the vocabulary'
            self.vocabulary = create_vocabulary(allWords,kmeans_conf,True)
            pickle.dump(self.vocabulary,open( dictionaryName, "wb" ))
        except:
            raise

        try:
            self.nn_data = pickle.load(open( nn_name, "rb" ))
            print 'Loaded NN graph! file name = '+nn_name
        except IOError:
            print 'Computing histograms and NN graph from training set'
            self.nn_data = self.trainNearestNeighbor(labeled['paths'],self.vocabulary,
                                                     labeled['numLabels'],numNeighbors,False)
            pickle.dump(self.nn_data,open( nn_name, "wb" ))
        except:
            raise

    def trainNearestNeighbor( self, paths , vocabulary, numLabels, numNeighbors , verbose=False ):
        labels = [image['label'] for image in paths]

        x = []
        for p in paths:
            filePath = p['path']
            if verbose:
                print 'training histogram for '+filePath
            x.append( compute_histogram(filePath,vocabulary,kmeans_conf,sift_conf))

        x = np.array(x)
        # nbrs = NearestNeighbors(n_neighbors=numNeighbors, algorithm='ball_tree').fit(x)
        nbrs = NearestNeighbors(n_neighbors=numNeighbors, algorithm='kd_tree').fit(x)

        return {'nbrs':nbrs,'labels':labels,'numLabels':numLabels}

    def classify(self, filePath):
        nbrs = self.nn_data['nbrs']
        labels = self.nn_data['labels']
        numLabels = self.nn_data['numLabels']

        h = compute_histogram(filePath,self.vocabulary,kmeans_conf,sift_conf)

        distance,indices = nbrs.kneighbors(h)

        hits = [0]*numLabels
        for idx,i in enumerate(indices.flat):
            # hits[labels[i]] = hits[labels[i]] + 1 # uniform weighting
            hits[labels[i]] = hits[labels[i]] + 1.0/(distance[0,idx]+0.2) # weight by distance

        return np.argmax(hits)

class BowSiftNN:
    def __init__(self, verbose=False):
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

        # brute force nearest-neighbor
        scores = []

        for h in self.histograms:
            candidate = h['histogram']
            scores.append((h['label'],np.sum(np.subtract(target,candidate)**2)))

        scores = sorted(scores,key=lambda x: x[1])

        N = min(numNeighbors,len(scores))
        hits = [0]*self.numLabels
        # TODO examine this to see what improves the results
        for i in xrange(N):
            # hits[scores[i][0]] += 1
            hits[scores[i][0]] += 1.0/(scores[i][1]+0.005)

        return np.argmax(hits)

# score the test set
labeledTraining = truncateFileList(findLabeledImages("../brown/data/train"),maxTrainingImages)
labeledTest = truncateFileList(findLabeledImages("../brown/data/test"),maxTrainingImages)

evaluateClassifier(BowSiftNN(True),labeledTraining,labeledTest)

print 'Done!'

# TODO explore "better" clustering on results.  Adjust K
