"""
Visualize fasttext embeddings (with T-sne)

"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import json
import pandas as pd
import bokeh.models as bm
import bokeh.plotting as pl
from bokeh.io import export_png, output_file
from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap, factor_mark, linear_cmap
from bokeh.palettes import Category20_9

# Load embeddings
embeddings = np.load(
    "data/models/output_matrix_fasttext.npy", allow_pickle=True)

# Load vocab (from Fasttext)
vocab = json.loads(open("data/vocabs/fasttext_vocab.json").read())
# Build the lookup-table
word2index = {word: index for index, word in enumerate(vocab)}

# Reduce embeddings dimension
tsne = TSNE(n_components=2, perplexity=27,
            n_iter=3000, verbose=3, metric="cosine")
embds_reduced = tsne.fit_transform(embeddings)
# Normalization
embds_reduced = StandardScaler().fit_transform(embds_reduced)

# Build dataframe


def make_dataframe(vectors_tsne, word2index):
    df = pd.DataFrame(data=vectors_tsne, columns=["x", "y"])
    df['word'] = list(word2index.keys())
    return df

# Draw plot


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


dataframe = make_dataframe(embds_reduced, word2index)
draw_tsne(df=dataframe, fp="data/assets/tsne_fasttext.html",
          alpha=0.9, show=True)
