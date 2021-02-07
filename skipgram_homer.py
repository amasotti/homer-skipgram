# -*- coding: utf-8 -*-
__author__ = 'Antonio Masotti'
__date__ = 'january 2021'

"""
All the main functions needed to train the model.
Defined here to make train_skipgram.py more readable

"""

# Imports
import json
import random

import matplotlib.pyplot as plt  # for loss plotting
import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
# intern imports
from utils.dataset import make_batch
from utils.utils import print_test

writer = SummaryWriter(comment="Testing",
                       log_dir="data/assets/")

# -------------------------------------------------------------------------
#                   LOAD RAW DATA AND CREATE DATASET
# -------------------------------------------------------------------------


def skip_gram_dataset(corpus, word2index, fp, window=5):
    """
    Given a corpus, a window_size and a dictionary with mappings word : index, it returns
    a long list of lists that can be used to train the Skip Gram version of the
    Word2Vec model
    """
    print("Creating Skipgram Dataset")
    dataset = []
    # loop over each sentence
    for sentence in tqdm(corpus, desc="Sententence in Corpus"):
        # take each word as target separately
        for center_word in range(len(sentence)):
            # loop in the window and be careful to not jump out of the boundaries :)
            for j in range(max(center_word - window, 0), min(center_word + window, len(sentence))):
                # jump the center word
                if j != center_word:
                    # append the context words in tuples
                    dataset.append(
                        [word2index[sentence[center_word]], word2index[sentence[j]]])
    np.save(fp, dataset, allow_pickle=True)
    return dataset


def load_corpus(fp):
    corpus = np.load(fp, allow_pickle=True)
    print("Corpus loaded ...")
    return corpus.tolist()


def load_dataset(fp):
    print("Loading Dataset ...")
    skipDataset = np.load(fp, allow_pickle=True)
    return skipDataset.tolist()


def load_vocab(fp):
    with open(fp, "r", encoding="utf-8") as vocab_json:
        vocab = json.load(vocab_json)
    return vocab


def lookup_tables(path):
    with open(path, "r", encoding="utf-8") as fp:
        word2index = json.load(fp)
    index2word = {i: w for w, i in word2index.items()}
    return word2index, index2word


# -------------------------------------------------------------------------
#                           TRAINING PHASE
# -------------------------------------------------------------------------
# For testing while training
TEST_WORDS = ["εἶμι", "θεός", "θεά", "ἔρχομαι",
              "ἔβην", "ἦλθε", "θυμόν", "γλαυκῶπις", "ἔρος"]


def switch_phase(dataset, params, vocab, train_bar, phase="train"):
    dataset.set_split(phase)
    print(
        f'DATASET SUBSET LOADED : {dataset._target_split} with size : {len(dataset)}')
    print('Whole Dataset size: ', dataset.data_size)
    print('Size of the vocabulary: ', len(vocab), '\n\n')

    Loader = make_batch(dataset=dataset,
                        device=params.device,
                        batch_size=params.batch,
                        shuffle=params.shuffle,
                        drop_last=params.drop_last)
    if phase == "train":
        train_bar.reset(total=dataset._target_size / params.batch)
    return Loader


def train_model(model, dataset, vocab, optimizer, scheduler, word2index, index2word, params, paths, plot=True):
    # Set progress bars
    epoch_bar = tqdm(desc="Epochs Routine",
                     total=params.epochs, position=0, leave=True)
    train_bar = tqdm(desc="Training phase", total=dataset.train_size /
                     params.batch, position=1, leave=True)

    # Loss
    losses_train = [4]
    losses_val = [4]
    best_loss = [4]
    batch_counter = 0  # as x_axis in tensorboard

    for epoch in trange(params.epochs):
        Loader = switch_phase(dataset=dataset, params=params,
                              vocab=vocab, phase="train", train_bar=train_bar)

        # Training batches
        for idx, (center, context) in enumerate(Loader):
            # Set training mode on
            model.train()

            loss = model(center, context)
            losses_train.append(loss.item())
            writer.add_scalar('train loss', loss.item(), batch_counter)
            batch_counter += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_postfix(epoch=epoch, loss=loss.item())
            train_bar.update()
            if idx % params.show_stats_after == 0:
                print_test(model=model, words=TEST_WORDS,
                           w2i=word2index, i2w=index2word, epoch=epoch, save=False, n=7, metrics="cosine")
                model.save(fp=paths.model, losses=losses_train,
                           check_loss=best_loss)
                model.save_embeddings(paths.embeddings)

        # Validation and lr adjustment
        Loader = switch_phase(dataset=dataset, params=params,
                              vocab=vocab, phase="val", train_bar=train_bar)
        val_bar = tqdm(desc="Validation phase", total=dataset.train_size /
                       params.batch, position=1, leave=True)

        # set eval mode on
        model.eval()
        for idx, (center, context) in enumerate(Loader):
            loss = model(center, context)
            losses_val.append(loss.item())
            writer.add_scalar('validation loss', loss.item(), batch_counter)
            batch_counter += 1
            scheduler.step(losses_val[-1])

            if idx % 200 == 0:
                val_bar.set_postfix(loss=loss.item(), epoch=epoch)
                val_bar.update(n=200)

        # after both train and val
        epoch_bar.update()

    if plot:
        plot = plot_loss(losses=losses_train, path=paths.plots)
    writer.close()

# ---------------------------------------------------------------------------------
#               PLOTS AND STATS
# ------------------------------------------------------------------------------------


def plot_some(data):
    if len(data) < 100:
        return data
    else:
        random_idx = []
        i = 0
        while i < 1000:
            r = random.randint(0, len(data)-1)
            if r not in random_idx:
                random_idx.append(r)
            i += 1
        return [data[j] for j in sorted(random_idx)]


def plot_loss(losses, path):
    plt.figure(figsize=(100, 100))
    plt.xlabel("batches")
    plt.ylabel("batch_loss")
    plt.title("loss vs #batch -- Training")
    plt.plot(plot_some(losses))
    plt.savefig(path)
    plt.show()
