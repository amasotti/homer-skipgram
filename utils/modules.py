"""

The torch.nn class

NN Model inspired by [n0obcoder](https://github.com/n0obcoder/Skip-Gram_Model-TensorFlow)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def debug(desc, tensor, show=False):
    '''
    Show the shape of each tensor if show=True

    '''
    if show:
        print(f"------------------------------------------------------------------------\n \
        {desc} : shape:{tensor.shape}\n------------------------------------------------------------------------------------\n\n")
    else:
        pass


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embeddings, device='cpu', negs=15, noise_dist=None):
        super(SkipGram, self).__init__()

        self.vocab_size = vocab_size
        self.embd_size = embeddings
        self.negs = negs
        self.device = device
        self.noise_dist = noise_dist
        self.embeddings_target = nn.Embedding(vocab_size, embeddings)
        self.embeddings_context = nn.Embedding(vocab_size, embeddings)

        self.initialize_embeddings(emb_size=embeddings)

    def initialize_embeddings(self, emb_size):
        custom_range = 0.5 / emb_size
        # Initialize embeddings for target words with values from the uniform distribution
        init.uniform_(self.embeddings_target.weight.data,  -
                      custom_range, custom_range)
        # Fill the tensor with the given value
        init.constant_(self.embeddings_context.weight.data, 0)

    def forward(self, target, context):
        # computing embeddings for target and context words
        # Target word embedding : size (b_size, emb)
        emb_input = self.embeddings_target(target)
        debug("embedding input", emb_input)

        # mask some terms in the input to prevent overfitting (https://github.com/keras-team/keras/issues/7290)
        # emb_input = F.dropout(emb_input, 0.1) #Commented for now, since I think it makes the performance slightly worse

        # Context word embedding (b_size, emb_dim)
        emb_context = self.embeddings_context(
            context)
        debug("embedding context", emb_context)

        # Multiply the two tensors together
        emb_together = torch.mul(emb_input, emb_context)  # b_size, emb_dim
        debug("First product of cont & input", emb_together)

        # Sum the values obtained (sum over embeddings, reduce to a 1d tensor)
        emb_together = torch.sum(emb_together, dim=1)  # batch_size

        # Apply softmax
        score = F.logsigmoid(emb_together)
        debug("score before negatives", score)  # bs

        # Now let's take care of the negatives
        if self.noise_dist is None:
            self.noise_dist = torch.ones(self.vocab_size)
        # Find out how many negative examples we need (here batch size * negs).
        negs_number = context.shape[0] * self.negs  # int, emb_dim * negative

        # build negs example (take random words from the corpus and adjust their weights)
        negative_example = torch.multinomial(
            self.noise_dist, negs_number, replacement=True)  # emb_dim * negatives (1d tensor)
        debug("negative_example", negative_example)

        # Move to cuda, without creating another tensor (viewes share the same underlying data with the copied tensors)
        negative_example = negative_example.view(
            context.shape[0], self.negs).to(self.device)  # bs, num_neg_samples

        # Calculate the embeddings (as context)
        emb_neg = self.embeddings_target(negative_example)
        debug("emb_neg", emb_neg)  # bs, neg, emb_dim

        neg_score = torch.bmm(
            emb_neg, emb_input.unsqueeze(2))  # b_size, negs, 1
        debug("neg_score after embedding", neg_score)

        neg_score = neg_score.squeeze(2).sum(1)
        debug("reduce dim of neg_score", neg_score)  # bsize

        neg_score = F.logsigmoid(neg_score)
        debug("neg_score logsigmoid", neg_score)

        return (-1 * (score + neg_score).mean())
