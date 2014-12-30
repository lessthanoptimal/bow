__author__ = 'pja'

from create_input_set import findLabeledImages
from math import *
import scipy
import random
import pickle

from vlfeat import *
import numpy as np

sift_conf = {'step':20,'fast':True}
kmeans_conf = {'K':50,'max_niters':100}
maxTrainingImages = -1


# Converts an image from integer to floating point format used by VLFeat
def convertImageToSingle(l):
    f = np.zeros(l.shape,dtype=numpy.float32)

    for i in range(l.shape[0]):
        for j in range(l.shape[1]):
            f[i,j] = l[i,j]/255.0
    return f

def compute_sift_words( filePath ):
    image = scipy.misc.imread(filePath)
    single = convertImageToSingle(image)
    return vl_dsift(single,step=sift_conf['step'],fast=sift_conf['fast'])

# Loads the entire training set and computes all the words in each image
def computeAllWords( paths , verbose=False):
    words = numpy.zeros((128,0),dtype=numpy.uint8)
    total = 0
    for p in paths:
        filePath = p['path']
        total += 1
        if verbose:
            print '%{: f}'.format(100.0*(total/float(len(paths))))

        [F,D]=compute_sift_words(filePath)

        words = numpy.hstack([words,D])

    return words

# the descriptor , set of all words (column vectors), which word is being scored
def score_match( desc , vocabulary , wordIndex ):
    score = np.sum(np.sqrt((desc-vocabulary[:,wordIndex])**2))/128.0
    return score
    # score = 0.0
    # for i in xrange(128):
    #     dx = desc[i] - words[i,wordIndex]
    #     score += sqrt(dx * dx)
    #
    # return score / 128.0

# score a set of k-means clusters
# @param dictionary The found k-means dictionary
# @param members List for each word in the dictionary which specified the words that are its members
# @param allWords set of all words which the dictionary was computed from
def score_kclusters( dictionary , memberships , allWords ):
    totalDistance = 0
    for i in memberships:
        d = dictionary[:,i]
        totalDistance += score_match(d,allWords,i)
    return totalDistance


# Use k-means clustering to find a simplified vocabulary from the set of words
def create_vocabulary( words , max_iterations=20 , verbose=False):

    if verbose:
        print 'total words = '+str(words.shape[1])

    K = kmeans_conf['K']
    best_vocabulary = numpy.zeros((128,K),dtype=numpy.int32)
    best_score = float('Inf')

    for i in xrange(max_iterations):
        [C, I] = vl_ikmeans(words,K=K,max_niters=kmeans_conf['max_niters'])
        score = score_kclusters(C,I,words)
        if score < best_score:
            if verbose:
                print 'Found better score '+str(score)+"  on iteration "+str(i)
            best_score = score
            best_vocabulary = C
        elif verbose:
            print 'Worse score on iteration '+str(i)

    return best_vocabulary

# Finds the work in the vocabulary which is the best for the target
def best_word( target , vocabulary , max_distance = float('inf')):
    best_word = -1
    best_score = float('inf')

    for i in xrange(vocabulary.shape[1]):
        score = score_match(target,vocabulary,i)
        if score < best_score and score <= max_distance:
            best_score = score
            best_word = i

    return best_word

def train_histogram( paths , vocabulary ):

    K = kmeans_conf['K']

    histograms = {}

    for p in paths:
        filePath = p['path']
        label = p['label']

        if label not in histograms:
            histograms[label] = [0]*K
        h = histograms[label]

        # compute the words for each image
        [F,D]=compute_sift_words(filePath)

        for i in xrange(D.shape[1]):
            bw = best_word(D[:,i],vocabulary)
            if bw >= 0:
                h[bw] += 1

    # normalize the histograms so that they sum up to one
    for key in histograms.keys():
        h = histograms[key]
        total = float(sum(h))
        histograms[key] = [x/total for x in h]

    return histograms

def classifyImage( filePath , vocabulary , histograms ):
    K = len(vocabulary)

    # find the words in the image
    [F,D]=compute_sift_words(filePath)

    # compute normalized histogram
    target = [0]*K
    for i in xrange(D.shape[1]):
        bw = best_word(D[:,i],vocabulary)
        if bw >= 0:
            target[bw] += 1
    total = float(sum(target))
    target = [x/total for x in target]

    # compare to labeled histograms
    best_match = -1
    best_score = 0

    for key in histograms.keys():
        score = np.multiply(target,histograms[key]).sum()
        if score > best_score:
            best_score = score
            best_match = key

    return best_match

labeledTraining = findLabeledImages("../brown/data/train")

# remove all but the first N images
if maxTrainingImages > -1 and maxTrainingImages < len(labeledTraining['paths']):
    random.shuffle(labeledTraining['paths'])
    labeledTraining['paths'] = labeledTraining['paths'][0:maxTrainingImages]

# Todo truncate

dictionaryName = 'dictionary{0:d}.p'.format(kmeans_conf['K'])
histogramsName = 'histograms{0:d}.p'.format(kmeans_conf['K'])

try:
    vocabulary = pickle.load(open( dictionaryName, "rb" ))
    print 'Loaded vocabulary! file name = '+dictionaryName
except IOError:
    print 'Computing words'
    allWords = computeAllWords(labeledTraining['paths'],True)
    print 'Creating the vocabulary'
    vocabulary = create_vocabulary(allWords,5,True)
    pickle.dump(vocabulary,open( dictionaryName, "wb" ))
    allWords = []
except:
    raise

try:
    histograms = pickle.load(open( histogramsName, "rb" ))
    print 'Loaded histogram! file name = '+histogramsName
except IOError:
    print 'Computing histograms from training set'
    histograms = train_histogram(labeledTraining['paths'],vocabulary)
    pickle.dump(histograms,open( histogramsName, "wb" ))
except:
    raise

# score the test set
labeledTest = findLabeledImages("../brown/data/test")

if maxTrainingImages > -1 and maxTrainingImages < len(labeledTest['paths']):
    random.shuffle(labeledTest['paths'])
    labeledTest['paths'] = labeledTest['paths'][0:maxTrainingImages]

total = 0
totalCorrect = 0
for p in labeledTest['paths']:
    filePath = p['path']
    label = p['label']
    print 'Scoring '+filePath
    found = classifyImage(filePath,vocabulary,histograms)
    total += 1
    if found is label:
        totalCorrect += 1
        # print'    correct! label = '+str(found)
    # else:
    #     print'    incorrect: expected '+str(label)+"  found "+str(found)

print 'Faction Correct: '+str(totalCorrect/float(total))


print 'Done!'

# TODO Compute histogram for all training images.  Save labeled results on a per image absis

# save results for the found vocabulary!!!