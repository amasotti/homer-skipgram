"""
Auxiliary functions

"""

import torch
import numpy as np


def save_model(model, epoch, losses, fp):
    """
    Compare the actual and the last loss value. If the value improved, save the model
    """
    if epoch > 0:  # wait at least 1 epoch
        print("Check if the model should be saved:")
        if losses[-1] < losses[-2]:
            print(
                f"Loss improved of {losses[-1]-losses[-2]} points, save the model")
            torch.save({'model_state_dict': model.state_dict(),
                        'losses': losses}, fp)
            return True
        else:
            print(
                f"Loss worsened by {losses[-1]-losses[-2]} points, skip saving")
            return False


# TEST
def nearest_word(target, embeddings, n=10):
    '''
    A kind of Projection formula, finds the closest vector in a vector space to the one given in input

    '''
    # calculate distance between target and embeddings
    # calc the distance of all vectors from the target
    distance = np.linalg.norm(target - embeddings, axis=1)

    # select the indx of the n closest
    idx_next_words = np.argsort(distance)[:n]

    # select only the vectors in the precedently found array
    distances = distance[idx_next_words]

    return idx_next_words, distances


def print_test(model, words, w2i, i2w, epoch):
    model.eval()
    emb_matrix = model.embeddings_target.weight.data.cpu()
    nearest_words_dict = {}

    print('\n==============================================\n')
    for w in words:
        # for each test word, select the corresponding embeddings from the model
        inp_emb = emb_matrix[w2i[w], :]
        # use the nearest_word function to get the indices of the closest words and the euclidean distance (later ignored)
        emb_ranking_top, _ = nearest_word(
            target=inp_emb, embeddings=emb_matrix)
        with open("data/assets/skipgram_predictions.txt", 'a', encoding="utf-8") as fp:
            fp.write(
                f"Epoch: {epoch}:\n{w.ljust(10)} |  {', '.join([i2w[i] for i in emb_ranking_top[1:]])}\n")

        # Print on the console for debug / curiosity
        print(w.ljust(10), ' | ', ', '.join(
            [i2w[i] for i in emb_ranking_top[1:]]))

    with open("data/assets/skipgram_predictions.txt", 'a', encoding="utf-8") as fp:
        fp.write(
            "\n----------------------------------------------------------------\n")

    return nearest_words_dict
