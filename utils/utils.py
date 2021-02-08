"""
Auxiliary functions

"""


import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd
import bokeh.models as bm
import bokeh.plotting as pl
from bokeh.io import export_png, output_file
from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap, factor_mark, linear_cmap
from bokeh.palettes import Category20_9

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


def tsne_reduction(embeddings, perplexity=20, metrics="euclidean"):
    scaler = StandardScaler()
    tsne = TSNE(n_components=2,
                perplexity=perplexity,
                metric=metrics,
                verbose=3,
                n_iter=500)
    vectors_tsne = tsne.fit_transform(embeddings)
    vectors_tsne = scaler.fit_transform(vectors_tsne)
    return vectors_tsne


def make_dataframe(vectors_tsne, word2index):
    df = pd.DataFrame(data=vectors_tsne, columns=["x", "y"])
    df['word'] = list(word2index.keys())
    return df


def draw_tsne(df, fp, alpha=0.69, width=1200, height=1000, show=True, title="Homer Embeddings"):
    """ draws an interactive plot for data points with auxilirary info on hover """
    output_file(fp, title=title)

    src = bm.ColumnDataSource(df)
    y = df['y']
    mapper = linear_cmap(
        field_name='y', palette=Category20_9, low=min(y), high=max(y))
    fig = pl.figure(active_scroll='wheel_zoom',
                    width=width, height=height, title=title)

    fig.scatter('x', 'y',
                size=10,
                line_color=mapper,
                color=mapper,
                fill_alpha=alpha,
                source=src)

    fig.add_tools(bm.HoverTool(tooltips=[("token", "@word")]))
    pl.save(fig, title=title)
    if show:
        pl.show(fig)
