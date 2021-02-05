"""

The torch.nn class

NN Model inspired by [n0obcoder](https://github.com/n0obcoder/Skip-Gram_Model-TensorFlow)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    def __init__(self, vocab_size, embeddings, device='cpu', noise_dist=None, negs=15,batch_size=4096):
        super(CBOW, self).__init__()

        self.vocab_size = vocab_size
        self.negs = negs
        self.device = device
        self.noise_dist = noise_dist

        self.embeddings_target = nn.Embedding(vocab_size, embeddings)
        self.embeddings_context = nn.Embedding(vocab_size, embeddings)

        self.initialize_embeddings(emb_size=embeddings)

    def initialize_embeddings(self, emb_size):
        custom_range = 0.5 / emb_size
        self.embeddings_target.weight.data.uniform_(-custom_range, custom_range)
        self.embeddings_context.weight.data.uniform_(-0, 0)


    def forward(self, target, context):
        # computing embeddings for target and context words
        emb_input = self.embeddings_target(target)  # bs, emb_dim (4096,100)
        emb_input = F.dropout(emb_input, 0.1) # mask some terms in the input to prevent overfitting (https://github.com/keras-team/keras/issues/7290)
        emb_context = self.embeddings_context(context)  # bs, emb_dim (4096,100)

        score = torch.mul(emb_input, emb_context)  # bs, emb_dim (4096,100)
        score = torch.sum(score, dim=1)  # bs
        loss = F.logsigmoid(-1 * score).squeeze()

        if self.negs > 0:
            # computing negative loss
            if self.noise_dist is None:
                self.noise_dist = torch.ones(self.vocab_size)

            # Find out how many negative examples we need (here batch size * negs).
            negs_number = context.shape[0] * self.negs
            # build negs example
            negative_example = torch.multinomial(self.noise_dist, negs_number,replacement=True)  # coz bs*num_neg_samples > vocab_size
            # Move to cuda, without creating another tensor (viewes share the same underlying data with the copied tensors)
            negative_example = negative_example.view(context.shape[0], self.negs).to(                self.device)  # bs, num_neg_samples

            # calculate the embds
            emb_negative = self.embeddings_context(negative_example)  # bs, neg_samples, emb_dim
            score = torch.bmm(emb_negative.neg(), emb_input.unsqueeze(2))  # bs, neg_samples, 1
            noise_loss = F.logsigmoid(score).squeeze(2).sum(1)  # bs
            loss_total = -(loss + noise_loss).mean()

            return loss_total
        else:
            raise ValueError('Negatives should be > 0, this is a Skip Gram Negative model')
