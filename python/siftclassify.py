from vlfeat import *
import numpy as np
import scipy
import random
import sklearn.neighbors # need so that scipy.misc works for some reason


# sift_conf = {'step':8,'size':4,'fast':True}
# kmeans_conf = {'K':50,'num_seeds':1,'max_niters':200}
# maxFeaturesPerImage = 150

# Converts an image from integer to floating point format used by VLFeat
def convertImageToSingle(l):
    f = np.zeros(l.shape,dtype=numpy.float32)

    for i in range(l.shape[0]):
        for j in range(l.shape[1]):
            f[i,j] = l[i,j]/255.0
    return f

def compute_sift_words( filePath , sift_conf):
    image = scipy.misc.imread(filePath)
    single = convertImageToSingle(image)
    return vl_dsift(single,step=sift_conf['step'],size=sift_conf['size'],fast=sift_conf['fast'])[1]

# Loads the entire training set and computes all the words in each image as one big set
def computeAllWords( paths , maxFeaturesPerImage, sift_conf, verbose=False):
    words = numpy.zeros((128,0),dtype=numpy.uint8)
    total = 0
    for p in paths:
        filePath = p['path']
        total += 1
        if verbose:
            print '%{: f}'.format(100.0*(total/float(len(paths))))

        D=compute_sift_words(filePath,sift_conf)

        # randomly select N of them
        N = min(D.shape[1],maxFeaturesPerImage)
        set=range(D.shape[1])
        random.shuffle(set)
        D = D[:,set[0:N]]

        # Add the new words onto the stack
        words = numpy.hstack([words,D])

    return words

# For each image in the path compute all the features and return them in a labeled set
def computeImageVLFeats( paths , maxFeaturesPerImage, sift_conf, verbose=False):
    output = []
    words = numpy.zeros((128,0),dtype=numpy.uint8)
    total = 0
    for p in paths:
        filePath = p['path']
        total += 1
        if verbose:
            print '%{: f}'.format(100.0*(total/float(len(paths))))

        D=compute_sift_words(filePath,sift_conf)

        # randomly select N of them
        if 0 < maxFeaturesPerImage < D.shape[1]:
            N = min(D.shape[1],maxFeaturesPerImage)
            set=range(D.shape[1])
            random.shuffle(set)
            D = D[:,set[0:N]]

        # Add the new words onto the stack
        output.append( {'label':p['label'],'features':D})

    return output

# the descriptor , set of all words (column vectors), which word is being scored
def score_match( desc , vocabulary , wordIndex ):
    # TODO potential area to tweak to improve efficiency.  only effects cluster selection
    score = np.sum(np.sqrt((desc-vocabulary[:,wordIndex])**2))/128.0
    # score = np.sum((desc-vocabulary[:,wordIndex])**2)/128.0
    return score
    # score = 0.0
    # for i in xrange(128):
    #     dx = desc[i] - vocabulary[i,wordIndex]
    #     score += dx*dx
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
def create_vocabulary( words , kmeans_conf, verbose=False):

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

def compute_histogram( filePath , vocabulary , kmeans_conf, sift_conf ):

    K = kmeans_conf['K']

    # find the words in the image
    D=compute_sift_words(filePath,sift_conf)

    # Much faster if the built in VL code is used here! Python equivalent is about 30x slower
    bestWords = vl_ikmeanspush(D,vocabulary)

    histogram = [0]*K
    for i in xrange(D.shape[1]):
        histogram[bestWords[i]] += 1

    # shitB = vl_ikmeanshist(K,shitA)

    # # TODO could compute histogram using distance here to improve accuracy?
    # # compute the histogram
    # histogram = [0]*K
    # for i in xrange(D.shape[1]):
    #     bw = best_word(D[:,i],vocabulary)
    #     if bw >= 0:
    #         histogram[bw] += 1
    # normalize the histogram so that it sums up to one
    total = float(sum(histogram))
    return [x/total for x in histogram]# todo any chance that normalizing the histogram is screwing up the results?
