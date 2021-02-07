"""
Auxiliary functions

"""

import numpy as np
import torch

# TEST


def nearest_word(target, embeddings, n=10, metrics="cosine"):
    """
    Return the n most similar words in the embedding matrix.
    Similarity is calculated as cosine similarity

    """
    if metrics == "cosine":
        # calculate cosine similarity between target word and each word in the embd matrix
        similarities = torch.cosine_similarity(target, embeddings, dim=-1)
        idx_most_similar = torch.topk(similarities, k=n).indices
        return idx_most_similar.tolist()
    elif metrics == "euclidean":
        distance = np.linalg.norm(target - embeddings, axis=1)
        # select the indx of the n closest
        idx_most_similar = np.argsort(distance)[:n]
        # select only the vectors in the precedently found array
        distances = distance[idx_most_similar]
        return idx_most_similar


def print_test(model, words, w2i, i2w, epoch, save=False, n=10, metrics="cosine"):
    model.eval()
    emb_matrix = model.emb_context.weight.data.cpu()
    print('\n==============================================\n')
    for w in words:
        try:
            # for each test word, select the corresponding embeddings from the model
            inp_emb = emb_matrix[w2i[w], :]
            # use the nearest_word function to get the indices of the closest words and the euclidean distance (later ignored)
            emb_ranking_top = nearest_word(
                target=inp_emb, embeddings=emb_matrix, n=n, metrics=metrics)
            if save:
                with open("data/assets/skipgram_predictions.txt", 'a', encoding="utf-8") as fp:
                    fp.write(
                        f"Epoch: {epoch}:\n{w.ljust(10)} |  {', '.join([i2w[i] for i in emb_ranking_top])}\n")

            # Print on the console for debug / curiosity
            print(w.ljust(10), ' | ', ', '.join(
                [i2w[i] for i in emb_ranking_top[1:]]))
        except KeyError:
            print("Word not found")
    if save:
        with open("data/assets/skipgram_predictions.txt", 'a', encoding="utf-8") as fp:
            fp.write("\n" + "="*20 + "\n")
