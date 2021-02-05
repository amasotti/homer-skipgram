# -*- coding: utf-8 -*-
__author__ = 'Antonio Masotti'
__date__ = 'january 2021'

# Imports
import json
import os
import random
from argparse import Namespace

import matplotlib.pyplot as plt  # for loss plotting
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm, trange

# intern imports
from utils.dataset import trainDataset, make_batch
from utils.modules import CBOW
from utils.utils import print_test, save_model

# popular words
paths = Namespace(
    # tokenized Ilias & Odyssey (list of lists, each list 1 sentence)
    corpus="./data/Homer_tokenized_corpus.npy",
    # Word to index lookup table
    word2index="./data/vocabs/Homer_word2index.json",
    # Dataset for SkipGram (list of ngrams : [target, context])
    skipDataset="./data/Homer_skipgram_dataset.npy",
    # Lookup word_with_frequencies (subsampled)
    vocab='./data/vocabs/Homer_word_frequencies.json',
    # Pytorch model
    #model='data/models/Skipgram_Pytorch_0502_beta.pth'
    model='data/models/Skipgram_Pytorch_0502_gamma.pth'
)

# -------------------------------------------------------------------------
#                   LOADING RAW DATA AND LOOK-UP TABLES
# -------------------------------------------------------------------------
# Load tokenized corpus (see preprocessing.py in utils)
corpus = np.load(paths.corpus, allow_pickle=True)
corpus = corpus.tolist()

# load the vocabulary (with subsampling)
with open(paths.vocab, "r", encoding="utf-8") as fp:
    vocab = json.load(fp)
print("Vocabulary successfully loaded")

# extract and save the dataset for training (only once, then load it)
# skipDataset = skip_gram_dataset(corpus=corpus, word2index=word2index, window=7)
# np.save(args.skipDataset, skipDataset,allow_pickle=True)

# Load the Word2Index dictionary (see preprocessing.py in utils))
with open(paths.word2index, "r", encoding="utf-8") as fp:
    word2index = json.load(fp)
print("Look-up dictionary successfully loaded")

# Create a reverse lookup table
index2word = {i: w for w, i in word2index.items()}

# Load tokenized corpus (see preprocessing.py in utils)
skipDataset = np.load(paths.skipDataset, allow_pickle=True)
skipDataset = skipDataset.tolist()
print("Dataset successfully loaded")

# -------------------------------------------------------------------------
#               SETTINGS FOR THE NEURAL MODEL
# -------------------------------------------------------------------------

TEST_WORDS = ['μηνιν', "εθηκε", "ερχομαι", "θεα", "υπνος",
              "βροτον", "ευχομαι", "ερος", "φατο", "εφατʼ", "βασιληα"]

params = Namespace(
    train_size=0.7,
    shuffle=False,  # TODO: Although it's a good idea to shuffle the batches, I have the problem that the whole dataset is shuffled per default and then I get a out of bound error
    drop_last=True,
    batch=1024 * 4,
    epochs=150,
    lr=0.001,  # automatically adjusted with the scheduler while training
    device='cpu',
    cuda=False,
    embeddings=100,
    show_stats_after=307,  # after how many batches should the bars be updated
)

if torch.cuda.is_available():
    params.cuda = True
    params.device = 'cuda'
else:
    params.cuda = False
    params.device = "cpu"

print(f"Using GPU ({params.device}) : {params.cuda}\n")

# Make Torch Dataset from list (splits data and transforms them into tensors)
Dataset = trainDataset(
    skipDataset, train_size=params.train_size)

# make noise distribution to sample negative examples from
word_freqs = np.array(list(vocab.values()))
unigram_dist = word_freqs / sum(word_freqs)
noise_dist = torch.from_numpy(
    unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))

model = CBOW(vocab_size=len(vocab),
             embeddings=params.embeddings,
             device=params.device,
             noise_dist=noise_dist,
             negs=15,
             batch_size=params.batch)

# Load model
saved = torch.load(os.path.join(paths.model))
model.load_state_dict(saved['model_state_dict'])
print('Model loaded successfully')

# move to cuda if available
model.to(params.device)

print('\nMODEL SETTINGS:')
print(model)

losses_train = [0] # loss each batch training
losses_val = [0] # loss each batch validation

optimizer = optim.RMSprop(model.parameters(), lr=params.lr, momentum=0.7,centered=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 mode="min",
                                                 factor=0.3, patience=1, verbose=True)
# Set bars

epoch_bar = tqdm(desc="Epochs Routine", total=params.epochs,
                 position=0, leave=True)
train_bar = tqdm(desc="Training phase", total=Dataset.train_size /
                 params.batch, position=2, leave=False)
val_bar = tqdm(desc="Validatation phase", total=Dataset.val_size /
               params.batch, position=2, leave=False)

# Make sure that the model is saved at least once
saved = False

# AND ..... GO .....
for epoch in trange(params.epochs):
    # Load specific splitted dataset
    Dataset.set_split('train')
    model.train()
    print(
        f'DATASET SUBSET LOADED : {Dataset._target_split} with size : {len(Dataset)}')
    print('Whole Dataset size: ', Dataset.data_size)
    print('Size of the vocabulary: ', len(vocab), '\n\n')

    Loader = make_batch(dataset=Dataset,
                        device=params.device,
                        batch_size=params.batch,
                        shuffle=False,
                        drop_last=params.drop_last)

    # Batch for the training phase
    for batch_idx, (inp, target) in enumerate(Loader):
        # Training modus (the test with the small word list requires setting the mode to eval)
        model.train()

        # reset gradients
        optimizer.zero_grad()
        loss = model(inp, target)

        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())

        # I want to know what are you doing...
        if batch_idx % params.show_stats_after == 0:
            # Run a small test
            print_test(model, TEST_WORDS, word2index, index2word, epoch=epoch)
        # update bar
        train_bar.set_postfix(loss=loss.item(), epoch=epoch)
        train_bar.update()

    saved = save_model(model=model, epoch=epoch,losses=losses_train, fp=paths.model)
    # Load specific splitted dataset
    Dataset.set_split('val')
    print(
        f'DATASET SUBSET LOADED : {Dataset._target_split} with size : {len(Dataset)}')
    print('Whole Dataset size: ', Dataset.data_size)
    print('Size of the vocabulary: ', len(vocab), '\n\n')

    Loader = make_batch(dataset=Dataset,
                        device=params.device,
                        batch_size=params.batch,
                        shuffle=params.shuffle,
                        drop_last=params.drop_last)

    # Evaluation / Validation mode
    model.eval()
    for batch_idx, (inp, target) in enumerate(Loader):

        loss = model(inp, target)
        losses_val.append(loss.item())
        scheduler.step(losses_val[-1])

        # I want to know what are you doing...
        if batch_idx % params.show_stats_after == 0:
            # Run a small test:
            print_test(model, TEST_WORDS, word2index, index2word, epoch=epoch)

        # update bar
        val_bar.set_postfix(loss=loss.item(), epoch=epoch)
        val_bar.update(n=params.show_stats_after)

    epoch_bar.update()

# If the model wasn't saved at all, save it now
if not saved:
    torch.save({'model_state_dict': model.state_dict(),
                'losses': losses_train}, paths.model)

def plot_some(data):
    if len(data) < 1000:
        return data
    else:
        random_idx = []
        for i in range(1000):
            r = random.randint(0, len(data))
            if r not in random_idx:
                random_idx.append(r)
    return [data[j] for j in sorted(random_idx)]


plt.figure(figsize=(100, 100))
plt.xlabel("batches")
plt.ylabel("batch_loss")
plt.title("loss vs #batch -- Training")
plt.plot(plot_some(losses_train))
plt.savefig('data/assets/losses_train.png')
plt.show()

plt.figure(figsize=(100, 100))
plt.xlabel("batches")
plt.ylabel("batch_loss")
plt.title("loss vs #batch -- Validation")

plt.plot(plot_some(losses_val))
plt.savefig('data/assets/losses_val.png')
plt.show()
