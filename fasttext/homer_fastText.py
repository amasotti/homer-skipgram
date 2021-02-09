"""
Light script to build a Fasttext model using the fasttext library

"""

import fasttext
import numpy as np
import json
from argparse import Namespace

# Paths
args = Namespace(
    raw_data="data/raw_data/HomerGesamt_cleaned.txt",
    model="data/models/homer-fastText.bin",
    vocab="data/vocabs/fasttext_vocab.json",
    embeddings="data/models/fasttext_embeddings.npy"
)

TEST_WORDS = ["λίσσομαι", "θεός", "θεά", "ἔρχομαι",
              "βαίνω", "θάλασσα", "θυμός", "ἔρος", "βούλομαι"]

model = fasttext.train_unsupervised(input=args.raw_data,
                                    model='skipgram',
                                    dim=300,
                                    ws=9,
                                    min_count=1,
                                    epoch=50)
model.save_model(args.model)

# Load the model, if already existent
#model = fasttext.load_model(args.model)

# testing:
for word in TEST_WORDS:
    print(word)
    print(model.get_nearest_neighbors(word))
    print("\n"*3)

# Extract the embeddings
output_matrix = model.get_output_matrix()
np.save(args.embeddings,
        output_matrix, allow_pickle=True)

# Extract the vocab
vocab = model.get_words()
with open(args.vocab, "w", encoding="utf-8") as fp:
    json.dump(vocab, fp, ensure_ascii=False)
