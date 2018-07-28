import json
import time
from gensim.models.keyedvectors import KeyedVectors
from pickle import load

def load_json(path="data/captions_train2014.json"):
    with open(path) as json_data:
        d = json.load(json_data)
    return d

def load_embeddings(path="data/glove.6B.50d.txt.w2v"):
    t0 = time.time()
    glove = KeyedVectors.load_word2vec_format(path, binary=False)
    t1 = time.time()
    print("elapsed %ss" % (t1 - t0))
    return glove

def load_stopwords(path="data/stopwords.txt"):
    with open(path, 'r') as r:
        stops = []
        for line in r:
            stops += [i.strip() for i in line.split('\t')]
    return stops

def load_resnet(path='data/resnet18_features_train.pkl'):
    data = load(open(path, 'rb'))
    return data