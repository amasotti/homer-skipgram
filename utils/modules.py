"""

The torch.nn class

NN Model inspired by [n0obcoder](https://github.com/n0obcoder/Skip-Gram_Model-TensorFlow)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class CBOW(nn.Module):
    def __init__(self, vocab_size, embeddings, device='cpu', negs=15, noise_dist=None):
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
        init.uniform_(self.embeddings_target.weight.data,  -custom_range, custom_range)
        init.constant_(self.embeddings_context.weight.data, 0)


    def forward(self, target, context):
        # computing embeddings for target and context words
        emb_input = self.embeddings_target(target)  # bs, emb_dim (4096,100)
        emb_input = F.dropout(emb_input, 0.1) # mask some terms in the input to prevent overfitting (https://github.com/keras-team/keras/issues/7290)
        emb_context = self.embeddings_context(context)  # bs, emb_dim (4096,100)

        if self.noise_dist is None:
            self.noise_dist = torch.ones(self.vocab_size)
        # Find out how many negative examples we need (here batch size * negs).
        negs_number = context.shape[0] * self.negs
        # build negs example
        negative_example = torch.multinomial(self.noise_dist, negs_number,replacement=True)  # coz bs*num_neg_samples > vocab_size
        # Move to cuda, without creating another tensor (viewes share the same underlying data with the copied tensors)
        negative_example = negative_example.view(context.shape[0], self.negs).to(self.device)  # bs, num_neg_samples
        # Calculate the embeddings
        emb_neg = self.embeddings_target(negative_example)

        score = torch.sum(torch.mul(emb_input, emb_context),dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -1 * F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg, emb_input.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -1 * torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, fp):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(fp, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
