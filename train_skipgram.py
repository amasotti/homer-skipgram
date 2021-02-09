"""
Actual training loop

inspired by [Pytorch implements Word2vec](https://programmer.group/pytorch-implements-word2vec.html)

"""
from utils.utils import draw_tsne, make_dataframe, tsne_reduction
from utils.dataset import trainDataset
import torch
import torch.optim as optim
import numpy as np
from argparse import Namespace
from utils.modules import SkipGram
from skipgram_homer import load_corpus, load_dataset, load_vocab, lookup_tables, skip_gram_dataset, train_model

# ---------------------------------------------------------
#                     ARGS AND PATHS
# ---------------------------------------------------------
# Paths
paths = Namespace(
    corpus="./data/Homer_tokenized_accented.npy",
    word2index="./data/vocabs/Homer_word2index_accented.json",
    skipDataset="./data/Homer_skipgram_dataset_accented.npy",
    vocab='./data/vocabs/Homer_word_frequencies_accented.json',
    # Pytorch model
    model='data/models/Skipgram_Pytorch_0602_delta.pth',
    embeddings="data/models/embeddings.npy",
    plots="data/assets/losses_plot.png",
    tsne_plot="data/assets/tsne_plot.html"
)

# Parameters for the Neural Network
params = Namespace(
    create_dataset=False,
    window=5,
    load_model=True,
    # Currently not using the validation set so much, but still useful to avoid overfitting and run the scheduler.
    train_size=0.90,
    # Already shuffled when creating the dataset (both training and validation)
    shuffle=False,
    drop_last=True,
    batch=2500,
    epochs=30,
    embeddings=250,
    neg_sample=7,
    lr=0.001,  # automatically adjusted with the scheduler while training
    lr_decay=0.97,
    device='cpu',
    cuda=False,
    show_stats_after=1500,  # after how many mini-batches should the progress bars be updated
    draw_tsne=True  # Pretty cool visualization but be aware that TSNE dim reduction could take several minuts for large tensors!
)

# Check if we can use the GPU
if torch.cuda.is_available():
    params.cuda = True
    params.device = 'cuda'
print(f"Using device {params.device} \t GPU found: {params.cuda}\n")


# ---------------------------------------------------------
#                      LOADINGS
# ---------------------------------------------------------

corpus = load_corpus(paths.corpus)
vocab = load_vocab(paths.vocab)
word2index, index2word = lookup_tables(paths.word2index)

if params.create_dataset:
    # first create the list
    dataset = skip_gram_dataset(corpus=corpus,
                                word2index=word2index,
                                window=params.window,
                                fp=paths.skipDataset)
    # Pytorch Dataset utils customized
    dataset = trainDataset(dataset, train_size=params.train_size)

else:
    dataset = load_dataset(paths.skipDataset)
    dataset = trainDataset(dataset, train_size=params.train_size)

# ---------------------------------------------------------
#                      NOISE DIST
# ---------------------------------------------------------

word_freqs = np.array(sorted(vocab.values(), reverse=True))
unigram_dist = word_freqs/word_freqs.sum()
noise_dist = torch.from_numpy(
    unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))
# ---------------------------------------------------------
#                      INITIALIZE MODEL
# ---------------------------------------------------------


model = SkipGram(vocab_size=len(vocab),
                 embeddings=params.embeddings,
                 device=params.device,
                 negs=params.neg_sample,
                 noise_dist=noise_dist.to(params.device),
                 batch_size=params.batch
                 )


# Load model if present
if params.load_model:
    # Load the model
    saved = torch.load(paths.model)
    model.load_state_dict(saved['model_state_dict'])
    print('Model loaded successfully')
else:
  # Create a new model
    print("Creating new model from scratch...\n")

# move to CUDA if possible
model.to(params.device)

optimizer = optim.Adam(model.parameters(), lr=params.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", threshold=1e-4, cooldown=2,
                                                 factor=params.lr_decay, patience=2)  # small decay if loss doesn't decrease

# ---------------------------------------------------------
#                     TRAINING
# ---------------------------------------------------------

train_model(model=model,
            dataset=dataset,
            params=params,
            paths=paths,
            vocab=vocab,
            optimizer=optimizer,
            scheduler=scheduler,
            word2index=word2index,
            index2word=index2word,
            plot=True)


# ---------------------------------------------------------
#                     VISUALIZATION
# ---------------------------------------------------------

if params.draw_tsne:
    embeddings = model.emb_context.weight.data.cpu()
    emb_tensors = tsne_reduction(
        embeddings, perplexity=15, metrics="euclidean")
    dataframe = make_dataframe(emb_tensors, word2index)
    draw_tsne(df=dataframe,
              fp=paths.tsne_plot,
              alpha=0.9,
              show=True
              )
