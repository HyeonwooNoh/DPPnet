## import packages for skipthoughts vectors
import os

import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy
import nltk

from collections import OrderedDict, defaultdict
from scipy.linalg import norm
from nltk.tokenize import word_tokenize

## import skipthoughts path
import sys
skipthoughts_path = '/media/hyeonwoonoh/TeraDrive/Projects' + \
                    '/cumulab/2015_SUMMER/002_image_qa/002_hyeonwoonoh/skip-thoughts'
print 'skipthoughts_path: %s' % skipthoughts_path
sys.path.append(skipthoughts_path)
import skipthoughts

## load table files
path_to_models = skipthoughts_path + '/models/'
print 'path_to_models: %s' % path_to_models

print 'loading skipthoughts dictionaries ..'
words = []
f = open(path_to_models + 'dictionary.txt', 'rb')
for line in f:
    words.append(line.decode('utf-8').strip())
f.close()
print 'done'
print ''

## load table
print 'loading uni and bi tables for skipthoughts ..'
utable = numpy.load(path_to_models + 'utable.npy')
print 'utable loading done'
btable = numpy.load(path_to_models + 'btable.npy')
print 'btable loading done'

utable = OrderedDict(zip(words, utable))
print 'utable rearranging done'
btable = OrderedDict(zip(words, btable))
print 'btable rearranging done'
print 'done'
print ''

# word dictionary check
d = defaultdict(lambda : 0)
for w in utable.keys():
    d[w] = 1

porting_data_path = './data/skipthoughts_porting/'
print 'porting_data_path: %s' % porting_data_path
vqa_vocab_path = porting_data_path + 'vqa_vocab.txt'
print 'vqa_vocab_path: %s' % vqa_vocab_path
print ''

print 'load vqa_vocab ..'
vqa_vocab = []
f = open(vqa_vocab_path, 'rb')
for line in f:
    vqa_vocab.append(line.decode('utf-8').strip())
f.close()
print 'done'
print ''

small_utable = numpy.zeros((len(vqa_vocab)+1, utable[utable.keys()[0]].shape[1]), \
                                                 dtype=utable[utable.keys()[0]].dtype)
small_btable = numpy.zeros((len(vqa_vocab)+1, btable[btable.keys()[0]].shape[1]), \
                                                 dtype=btable[btable.keys()[0]].dtype)
print 'vqa vocab size: %d' % len(vqa_vocab)
print 'small_utable size: %d x %d ' % (small_utable.shape[0], small_utable.shape[1])
print 'small_btable size: %d x %d ' % (small_btable.shape[0], small_btable.shape[1])
print ''

print 'copying vqa vocab vectors to small utable and btable ..'
for i, v in enumerate(vqa_vocab):
    if d[v] > 0:
        small_utable[i] = utable[v]
        small_btable[i] = btable[v]
    else:
        small_utable[i] = utable['UNK']
        small_btable[i] = btable['UNK']
print 'last vector is for <eos>'
small_utable[len(vqa_vocab)] = utable['<eos>']
small_btable[len(vqa_vocab)] = btable['<eos>']
print 'done'
print ''

print 'save small_utable and small_btable to file'
small_utable_path = porting_data_path + 'vqa_utable.npy'
small_btable_path = porting_data_path + 'vqa_btable.npy'
print 'vqa utable path: %s' % small_utable_path
print 'vqa btable path: %s' % small_btable_path
print 'saving ..'
numpy.save(small_utable_path, small_utable)
numpy.save(small_btable_path, small_btable)
print 'done'
print ''


       

















