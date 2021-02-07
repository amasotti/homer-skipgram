"""

The torch.nn class

NN Model inspired by [n0obcoder](https://github.com/n0obcoder/Skip-Gram_Model-TensorFlow)

"""

import numpy as np
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
    def __init__(self, vocab_size, embeddings, batch_size, device='cpu', noise_dist=None, negs=15):
        super(SkipGram, self).__init__()
        self.noise_dist = noise_dist
        self.vocab_size = vocab_size
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.negs = negs
        self.device = device

        # define embeddings
        self.emb_center = nn.Embedding(self.vocab_size, self.embeddings)
        self.emb_context = nn.Embedding(self.vocab_size, self.embeddings)

        self.init_emb()

    def init_emb(self):
        custom_range = 0.5 / self.embeddings
        init.uniform_(self.emb_center.weight.data, -custom_range, custom_range)
        init.uniform_(self.emb_context.weight.data, -
                      custom_range, custom_range)

    def create_embeddings(self, center, context):
        # Embedding for input word
        emb_center = self.emb_center(center)  # batch, emb
        emb_center = F.dropout(emb_center, p=0.1, training=True)
        debug("emb_center", emb_center)

        # Embedding for context
        emb_context = self.emb_context(context)  # batch, emb
        debug("emb_context", emb_context)

        # embeddings for noise_dist
        if self.noise_dist is None:
            self.noise_dist = torch.ones(self.vocab_size)  # vocab_size
        debug("noise dist", self.noise_dist)

        # Take the noise_dist of size vocab_size and select from the multinomial dist batch_size * negs indices
        noise_dist = torch.multinomial(
            self.noise_dist, self.batch_size * self.negs, replacement=True)
        noise_dist.to(self.device)
        debug("Noise after torch.multinomial", noise_dist)

        emb_neg = self.emb_context(noise_dist).view(
            self.batch_size, self.negs, self.embeddings)  # batch, negs, emb_size
        debug("Embedding negative, size adjusted", emb_neg)

        return emb_center, emb_context, emb_neg

    def forward(self, center, context):

        emb_center, emb_context, emb_noise = self.create_embeddings(
            center, context)

        # Build the 3d tensors: center and context are batch_size, embd_size + 1d unsqueezed, noise_dist is batch_size, negs, embd_size
        emb_center = emb_center.view(
            self.batch_size, self.embeddings, 1)  # batch_size, emb_dim, 1
        debug("emb_center after View (forward)", emb_center)
        emb_context = emb_context.view(
            self.batch_size, 1, self.embeddings)  # batch_size, 1, emb_dim
        debug("emb_context after View", emb_context)

        # Matrix multiplication for true context and center word
        # batch, 1, 1 (one prediction per input in batch)
        true_loss = torch.bmm(emb_context, emb_center)
        debug("Context loss after BMM", true_loss)
        # apply sigmoid and remove 1s
        # size of true_loss : batch_size (the probable context words for the given emb_center)
        true_loss = F.logsigmoid(true_loss).squeeze()
        debug("Logsigmoid(context_loss)", true_loss)

        # Matrix multiplication for negative sampling (with sum)
        # noise_loss initialized :  batch_size, negs, 1
        # we want to have negs vectors for each input in batch_size
        noise_loss = torch.bmm(emb_noise.neg(), emb_center)
        debug("Noise loss after BMM", noise_loss)
        noise_loss = F.logsigmoid(noise_loss).squeeze()  # batch_size, negs
        debug("Logsigmoid(noise_loss)", noise_loss)
        # batch_size (1 prediction per input in batch)
        noise_loss = noise_loss.sum(1)
        debug("Noise loss summed up", noise_loss)

        loss_total = -(true_loss + noise_loss).mean()  # final loss
        debug("Final loss for this batch:", loss_total)

        return loss_total

    def save_embeddings(self, fp):
        embs = self.emb_context.weight.cpu().data.numpy()
        np.save(fp, embs, allow_pickle=True)

    def save(self, fp, losses, check_loss):
        # If the actual loss (in the training) has improved save it in the checklist and save the model
        if losses[-1] < check_loss[-1]:
            print(
                f"Actual loss: {losses[-1]}\nLast best loss: {check_loss[-1]}")
            print("\nSaving the model model\n")
            check_loss.append(losses[-1])
            torch.save(
                {'model_state_dict': self.state_dict(), 'losses': losses}, fp)
        else:
            print("Loss didn't improve, skip saving")
