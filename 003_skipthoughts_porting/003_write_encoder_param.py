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

print 'save uni, bi GRU encoder parameter for torch porting'
print ''

## import skipthoughts path
import sys
skipthoughts_path = '/media/hyeonwoonoh/TeraDrive/Projects' + \
                    '/cumulab/2015_SUMMER/002_image_qa/002_hyeonwoonoh/skip-thoughts'
print 'skipthoughts_path: %s' % skipthoughts_path
sys.path.append(skipthoughts_path)
import skipthoughts

## load model options
print 'Loading model parameters ...'
path_to_models = skipthoughts_path + '/models/'
path_to_umodel = path_to_models + 'uni_skip.npz'
path_to_bmodel = path_to_models + 'bi_skip.npz'
print '   path_to_models: %s' % path_to_models
print '   path_to_umodel: %s' % path_to_umodel
print '   path_to_bmodel: %s' % path_to_bmodel
print ''

print 'read uni, bi model options'
with open('%s.pkl'%path_to_umodel, 'rb') as f:
    uoptions = pkl.load(f)
with open('%s.pkl'%path_to_bmodel, 'rb') as f:
        boptions = pkl.load(f)
print 'done'
print ''

print 'init and load uparams'
uparams = skipthoughts.init_params(uoptions)
uparams = skipthoughts.load_params(path_to_umodel, uparams)
print(uparams.keys())
print 'done'
print ''

print 'init and load bparams'
bparams = skipthoughts.init_params_bi(boptions)
bparams = skipthoughts.load_params(path_to_bmodel, bparams)
print(bparams.keys())
print 'done'
print ''

print 'start saving'
porting_data_path = './data/skipthoughts_porting/'
print 'porting_data_path: %s' % porting_data_path
print ''

print 'save_uparams'
for k in uparams.keys():
    save_path = porting_data_path + 'uparams_%s.npy' % k
    print '   saving [%s] ..' % save_path,
    numpy.save(save_path, uparams[k]) 
    print '  done'
print ''

print 'save_bparams'
for k in bparams.keys():
    save_path = porting_data_path + 'bparams_%s.npy' % k
    print '   saving [%s] ..' % save_path,
    numpy.save(save_path, bparams[k]) 
    print '  done'
print ''


print 'done'
print ''


