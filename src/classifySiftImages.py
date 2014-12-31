__author__ = 'pja'

from create_input_set import findLabeledImages
from math import *
from sklearn.neighbors import NearestNeighbors
import scipy
import random
import pickle

from vlfeat import *
import numpy as np

sift_conf = {'step':8,'size':4,'fast':True}
kmeans_conf = {'K':400,'num_seeds':1,'max_niters':200}
maxTrainingImages = -1
maxFeaturesPerImage = 150
# number of neighbors in NN classifier
numNeighbors = 10


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
    return vl_dsift(single,step=sift_conf['step'],size=sift_conf['size'],fast=sift_conf['fast'])

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

        # randomly select N of them
        N = min(D.shape[1],maxFeaturesPerImage)
        set=range(D.shape[1])
        random.shuffle(set)
        D = D[:,set[0:N]]

        # Add the new words onto the stack
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
def create_vocabulary( words , verbose=False):

    if verbose:
        print 'total words = '+str(words.shape[1])

    K = kmeans_conf['K']
    best_vocabulary = numpy.zeros((128,K),dtype=numpy.int32)
    best_score = float('Inf')

    num_seeds = kmeans_conf['num_seeds']

    for i in xrange(num_seeds):
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

# Finds the word in the vocabulary which is the best for the target
def best_word( target , vocabulary , max_distance = float('inf')):
    best_word = -1
    best_score = float('inf')

    for i in xrange(vocabulary.shape[1]):
        score = score_match(target,vocabulary,i)
        if score < best_score and score <= max_distance:
            best_score = score
            best_word = i

    return best_word

def compute_histogram( filePath , vocabulary ):
    K = kmeans_conf['K']

    # find the words in the image
    [F,D]=compute_sift_words(filePath)

    # TODO could compute histogram using distance here to improve accuracy?
    # compute the histogram
    histogram = [0]*K
    for i in xrange(D.shape[1]):
        bw = best_word(D[:,i],vocabulary)
        if bw >= 0:
            histogram[bw] += 1
    # normalize the histogram so that it sums up to one
    total = float(sum(histogram))
    return [x/total for x in histogram]

def trainNearestNeighbor( paths , vocabulary, numLabels, numNeighbors , verbose=False ):
    labels = [image['label'] for image in paths]

    x = []
    for p in paths:
        filePath = p['path']
        if verbose:
            print 'training histogram for '+filePath
        x.append( compute_histogram(filePath,vocabulary))

    x = np.array(x)
    # nbrs = NearestNeighbors(n_neighbors=numNeighbors, algorithm='ball_tree').fit(x)
    nbrs = NearestNeighbors(n_neighbors=numNeighbors, algorithm='kd_tree').fit(x)

    return {'nbrs':nbrs,'labels':labels,'numLabels':numLabels}

def classifyNearestNeighbor( filePath , vocabulary , classifierData ):
    nbrs = classifierData['nbrs']
    labels = classifierData['labels']
    numLabels = classifierData['numLabels']

    h = compute_histogram(filePath,vocabulary)

    distance,indices = nbrs.kneighbors(h)

    hits = [0]*numLabels
    for idx,i in enumerate(indices.flat):
        # hits[labels[i]] = hits[labels[i]] + 1 # uniform weighting
        hits[labels[i]] = hits[labels[i]] + 1.0/(distance[0,idx]+0.2) # weight by distance

    return np.argmax(hits)

labeledTraining = findLabeledImages("../brown/data/train")

# remove all but the first N images
if maxTrainingImages > -1 and maxTrainingImages < len(labeledTraining['paths']):
    random.shuffle(labeledTraining['paths'])
    labeledTraining['paths'] = labeledTraining['paths'][0:maxTrainingImages]

# Todo truncate

dictionaryName = 'dictionary{0:d}.p'.format(kmeans_conf['K'])
nn_name = 'nndata{0:d}_NN{1:d}.p'.format(kmeans_conf['K'],numNeighbors)

try:
    vocabulary = pickle.load(open( dictionaryName, "rb" ))
    print 'Loaded vocabulary! file name = '+dictionaryName
except IOError:
    print 'Computing words'
    allWords = computeAllWords(labeledTraining['paths'],True)
    print 'Creating the vocabulary'
    vocabulary = create_vocabulary(allWords,True)
    pickle.dump(vocabulary,open( dictionaryName, "wb" ))
    allWords = []
except:
    raise

try:
    nn_data = pickle.load(open( nn_name, "rb" ))
    print 'Loaded NN graph! file name = '+nn_name
except IOError:
    print 'Computing histograms and NN graph from training set'
    nn_data = trainNearestNeighbor(labeledTraining['paths'],vocabulary,labeledTraining['numLabels'],numNeighbors,False)
    pickle.dump(nn_data,open( nn_name, "wb" ))
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
    found = classifyNearestNeighbor(filePath,vocabulary,nn_data)
    total += 1
    if found == label:
        totalCorrect += 1
        print'    correct! label = '+str(found)
    else:
        print'    incorrect: expected '+str(label)+"  found "+str(found)

print 'Faction Correct: '+str(totalCorrect/float(total))


print 'Done!'

# TODO Benchmark KD-Tree versus brute force
# TODO explore "better" clustering on results.  Adjust K
# TODO Create a generic class for classification
# TODO Linear SVM classifier
# TODO create a confusion matrix from results

# save results for the found vocabulary!!!