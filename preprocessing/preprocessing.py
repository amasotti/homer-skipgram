"""
Build vocab & corpus and save them for later use

"""

__author__ = "Antonio Masotti"
__date__ = "Febrauar 2021"

import json
from argparse import Namespace
from cltk.tokenize.sentence import TokenizeSentence
from cltk.tokenize.word import WordTokenizer
from tqdm import tqdm

from utils.utils import *
# Paths (I hate to have constantly to write paths...)
args = Namespace(
    raw_data="../data/raw_data/HomerGesamt_deaccented.txt",
    word2index="../data/vocabs/Homer_word2index.json",
    word_frequencies="../data/vocabs/Homer_word_frequencies.json"
)

# Load data
with open(args.raw_data, 'r', encoding='utf-8') as src:
    data = src.read()


def createCorpus(text, save=True):
    '''
    :params text - the raw text

    returns  + the corpus, a list of list with tokenized sentences
             + the vocab (a dictionary with the frequency of the tokens scaled by the total number of words.

    '''
    # load stopwords
    with open('../../data/stopwords.txt', 'r', encoding="UTF-8") as src:
        stopwords = src.read()

    # add punctuation signs
    stopwords = stopwords.split('\n')
    stopwords.extend([".", ",", "?", "!", "-", ":", ";", "·"])

    # tokenize sentences and then words
    Stokenizer = TokenizeSentence('greek')
    Wtokenizer = WordTokenizer('greek')

    sentences = Stokenizer.tokenize(text)
    new_sentences = []
    vocab = dict()

    print('Building corpus and freqDictionary')
    # for each sentence
    for sent in tqdm(sentences, desc="Sentences"):
        # extract the words
        new_sent = Wtokenizer.tokenize(sent)
        # Stopword deletion
        new_sent = [w for w in new_sent if w not in stopwords]
        new_sentences.append(new_sent)
        # add each word to dictionary or update count
        for w in new_sent:
            if w not in vocab:
                vocab[w] = 1
            else:
                vocab[w] += 1
    vocab_size = len(vocab)

    # Subsampling, see paper by Goldberg & Levy
    for k, v in vocab.items():
        frac = v / vocab_size
        p_w = (1+np.sqrt(frac * 0.001)) * 0.001 / frac
        # update the value for the word
        vocab[k] = p_w

    if save:
        print('Saving the frequencies')
        with open(args.word_frequencies, 'w', encoding='utf-8') as fp:
            json.dump(vocab, fp, ensure_ascii=False)

        print('Saving the corpus')
        arr = np.array(new_sentences, dtype=object)
        np.save('../../data/Homer_tokenized_corpus.npy', arr)

    return new_sentences, vocab


# tokenize and build corpus and freqDict from text
corpus, freqVocab = createCorpus(data)

# Lookup tables
word2index = {w: i for i, w in enumerate(freqVocab.keys())}

# save lookup dictionary
with open(args.word2index, 'w', encoding='utf-8') as fp:
    json.dump(word2index, fp, ensure_ascii=False)
