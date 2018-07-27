# STUDENT CODE
# define the punctuation-removal function
import re, string
from math import log10
import numpy as np

# this creates a regular expression that identifies all punctuation character
# don't include this in `strip_punc`, otherwise you will re-compile this expression
# every time you call the function
punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

def strip_punc(corpus):
    """ Removes all punctuation from a string.

        Parameters
        ----------
        corpus : str

        Returns
        -------
        str
            the corpus with all punctuation removed"""
    # substitute all punctuation marks with ""
    return punc_regex.sub('', corpus)

def tokenize(text):
    """ Splits the text into a lits of tokens
    
        Parameters
        ----------
        text: str
        
        Returns
        -------
        tokens: list of str"""
    return text.split()

from collections import Counter

def reduce_captions(captions, glove, stops=[]):
    """Takes a list of strings, representing captions
    Returns a list of shape (50,) ndarrays"""
    sequences = {Id: tokenize(strip_punc(caption.lower())) for Id, caption in captions.items()}
    for sequence in sequences.values():
        for i in range(len(sequence)-1, -1, -1):
            if sequence[i] in stops:
                del sequence[i]
    counters = [Counter(sequence) for sequence in sequences.values()]
    
    out = {}
    for Id, sequence in sequences.items():
        seq = []
        for word in sequence:
            if word in glove.vocab:
                seq.append(glove[word]*log10(len(counters) / sum(word in counter for counter in counters)))
        out[Id] = np.mean(seq, axis=0)
    
    return out