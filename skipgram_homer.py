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
from utils.dataset import trainDataset, make_batch, skip_gram_dataset
from utils.modules import SkipGram
from utils.utils import print_test, save_model

# Paths
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
    model='data/models/Skipgram_Pytorch_0602_delta.pth',
    # model='data/models/Skipgram_Pytorch_0502_gamma.pth',
    embeddings="data/models/embeddings.txt"
)
# Parameters for the Neural Network
params = Namespace(
    load_model=False,
    # Currently not using the validation set so much, but still useful to avoid overfitting and run the scheduler.
    train_size=0.90,
    # Already shuffled when creating the dataset (both training and validation)
    shuffle=False,
    drop_last=True,
    batch=1000,
    epochs=200,
    lr=0.001,  # automatically adjusted with the scheduler while training
    device='cpu',
    cuda=False,
    embeddings=250,
    show_stats_after=1500,  # after how many mini-batches should the progress bars be updated
)

# For testing while training
TEST_WORDS = ['μηνιν', "εθηκε", "ερχομαι", "θεα", "υπνος",
              "βροτον", "ευχομαι", "ερος", "φατο", "εφατʼ", "βασιληα"]

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

# Load the Word2Index dictionary (see preprocessing.py in utils))
with open(paths.word2index, "r", encoding="utf-8") as fp:
    word2index = json.load(fp)
print("Look-up dictionary successfully loaded")

# Create a reverse lookup table
index2word = {i: w for w, i in word2index.items()}

# extract and save the dataset for training (only once, then load it)
#skipDataset = skip_gram_dataset(corpus=corpus, word2index=word2index, window=5)
#np.save(paths.skipDataset, skipDataset,allow_pickle=True)

# Load tokenized corpus (see preprocessing.py in utils)
skipDataset = np.load(paths.skipDataset, allow_pickle=True)
skipDataset = skipDataset.tolist()
print("Dataset successfully loaded")

# -------------------------------------------------------------------------
#               SETTINGS FOR THE NEURAL MODEL
# -------------------------------------------------------------------------

# Check if we can use the GPU
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

# -------------------------------------------------------------------------
#                           INITIALIZE MODEL
# -------------------------------------------------------------------------
model = SkipGram(vocab_size=len(vocab),
                 embeddings=params.embeddings,
                 device=params.device,
                 negs=50,
                 noise_dist=None  # train without noising at beginning, then you can add noise
                 )

# Load model if present
if params.load_model:
    saved = torch.load(os.path.join(paths.model))
    model.load_state_dict(saved['model_state_dict'])
    print('Model loaded successfully')
else:
    print("Creating new model from scratch...")

# move to cuda if available
model.to(params.device)

print('\nMODEL SETTINGS:')
print(model)

# Optimizer and scheduler
optimizer = optim.Adamax(model.parameters(), lr=params.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", threshold=1e-4, cooldown=2,
                                                 factor=0.95, patience=2, min_lr=1e-09)  # small decay if loss doesn't decrease
# Set progress bars
epoch_bar = tqdm(desc="Epochs Routine", total=params.epochs,
                 position=0, leave=True)
train_bar = tqdm(desc="Training phase", total=Dataset.train_size /
                 params.batch, position=0, leave=True)


# Lists for keeping trace of the losses
losses_train = [0]  # loss each batch training
losses_val = [0]  # loss each batch validation
# look at this list and decide if the model improved or not
# start saving if the loss is smaller than this initial value
losses_save = [2e-01]

# AND ..... GO .....


# -------------------------------------------------------------------------
#                           TRAINING PHASE
# -------------------------------------------------------------------------
for epoch in trange(params.epochs):
    # Load specific splitted dataset
    Dataset.set_split('train')
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
    train_bar.reset(total=Dataset._target_size / params.batch)
    print("\n")
    for batch_idx, (inp, target) in enumerate(Loader):
        # Training modus (the test with the small word list requires setting the mode to eval)
        model.train()

        # reset gradients
        optimizer.zero_grad()

        loss = model(inp, target)
        losses_train.append(loss.item())

        loss.backward()
        optimizer.step()

        # I want to know what are you doing...
        if batch_idx % params.show_stats_after == 0:
            # Run a small test
            print_test(model, TEST_WORDS, word2index, index2word, epoch=epoch)
        # update bar
        if batch_idx % 200 == 0:
            save_model(model=model, epoch=epoch,
                       losses=losses_save, actual_loss=loss.item(), fp=paths.model)
            # Print actual learning rate
            for group in optimizer.param_groups:
                try:
                    actual_lr = group['lr']
                except:
                    actual_lr = 'unknown'
            print("\n")
            train_bar.set_postfix(
                loss=loss.item(), epoch=epoch, lr=actual_lr)
            train_bar.update(n=200)
            print("\n")

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

    # Set validation bar
    val_bar = tqdm(desc="Validatation phase", total=(
        Dataset._target_size / params.batch), position=0, leave=True)

    for batch_idx, (inp, target) in enumerate(Loader):

        loss = model(inp, target)
        losses_val.append(loss.item())
        scheduler.step(losses_val[-1])

        # I want to know what are you doing...
        # if batch_idx % params.show_stats_after == 0:
        # Run a small test:
        #print_test(model, TEST_WORDS, word2index, index2word, epoch=epoch)

        if batch_idx % 100 == 0:
            # update bar
            val_bar.set_postfix(loss=loss.item(), epoch=epoch)
            val_bar.update(n=100)

    epoch_bar.update()

# save as npy
embeddings = model.embeddings_target.weight.cpu().data.numpy
np.save("./data/models/embeddings.npy", embeddings, allow_pickle=True)


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
