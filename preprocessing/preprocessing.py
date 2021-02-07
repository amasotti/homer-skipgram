"""
Build vocab & corpus and save them for later use

"""

__author__ = "Antonio Masotti"
__date__ = "Febrauar 2021"

import json
import numpy as np
import math
import random
from argparse import Namespace
from cltk.tokenize.sentence import TokenizeSentence
from cltk.tokenize.word import WordTokenizer
from numpy.testing._private.utils import decorate_methods
from tqdm import tqdm

# Paths (I hate to have constantly to write paths...)
args = Namespace(
    raw_data="../data/raw_data/HomerGesamt_cleaned.txt",
    word2index="../data/vocabs/Homer_word2index_accented.json",
    word_frequencies="../data/vocabs/Homer_word_frequencies_accented.json",
    subsampling_vocab="../data/vocabs/Homer_subsampled.json"
)


def createCorpus(text, save=True):
    '''
    :params text - the raw text

    returns  + the corpus, a list of list with tokenized sentences
             + the vocab (a dictionary with the frequency of the tokens scaled by the total number of words.

    '''
    # load stopwords
    with open('../data/stopwords.txt', 'r', encoding="UTF-8") as src:
        stopwords = src.read()

    # add punctuation signs
    stopwords = stopwords.split('\n')
    stopwords.extend([".", ",", "?", "!", "-", ":",
                      ";", "·", "”", "“", "«", "»"])

    # tokenize sentences and then words
    Stokenizer = TokenizeSentence('greek')
    Wtokenizer = WordTokenizer('greek')

    sentences = Stokenizer.tokenize(text)
    new_sentences = []
    vocab = dict()

    print('Building corpus and freqDictionary')
    total_tokens = 0
    check = 0
    # for each sentence
    for sent in tqdm(sentences, desc="Sentences"):
        # extract the words
        new_sent = Wtokenizer.tokenize(sent)
        check += len(new_sent)
        # Stopword deletion
        new_sent = [w for w in new_sent if w not in stopwords]
        new_sentences.append(new_sent)
        total_tokens += len(new_sent)
        # add each word to dictionary or update count
        for w in new_sent:
            # Increment tokens count
            if w not in vocab:
                vocab[w] = 1
            else:
                vocab[w] += 1
    vocab_size = len(vocab)

    print("total tokens: ", total_tokens)
    print("total token (incl. stopwords)", check)
    print("vocab_size : ", vocab_size)
    # Subsampling
    treshold = 10e-05
    for k, v in vocab.items():
        # http: // mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        # Not really used for subsampling here but to generate the noise distribution
        frac = v / total_tokens
        p_w = (1 + math.sqrt(frac/treshold)) * (treshold/frac)
        vocab[k] = p_w

    if save:
        print('Saving the frequencies')
        with open(args.word_frequencies, 'w', encoding='utf-8') as fp:
            json.dump(vocab, fp, ensure_ascii=False)

        print('Saving the corpus')
        arr = np.array(new_sentences, dtype=object)
        np.save('../data/Homer_tokenized_accented.npy', arr)

        with open('../data/vocabs/Homer_wordList.csv', "w", encoding="utf-8") as fp:
            for idx, word in tqdm(enumerate(vocab)):
                fp.write(str(idx) + "," + word + "\n")

    return new_sentences, vocab


if __name__ == '__main__':
    import os
    # Load data
    with open(args.raw_data, 'r', encoding='utf-8') as src:
        data = src.read()
    corpus, freq_vocab = createCorpus(text=data, save=True)

    # Lookup tables
    word2index = {w: i for i, w in enumerate(freq_vocab.keys())}

    # save lookup dictionary
    with open(args.word2index, 'w', encoding='utf-8') as fp:
        json.dump(word2index, fp, ensure_ascii=False)
