from nem.util import load_songs
from openTSNE import TSNE, affinity, initialization, TSNEEmbedding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from absl import app, flags

seed = 12261532


def main(vargs):
    songs = load_songs('songs_filtered.csv')

    songs.tempo = songs.tempo / 140
    songs.loudness = (songs.loudness - songs.loudness.min()) / (songs.loudness.max() - songs.loudness.min())

    # only 6 most genres
    genre_counts = pd.Series(songs.metagenre).value_counts()
    genres_to_keep = genre_counts.iloc[1:7]
    print(genres_to_keep)


    features = [
        'danceability',
        'energy',
        #'key',
        #'loudness',
        #'mode',
        'speechiness',
        'acousticness',
        'instrumentalness',  # extremly discriminative!
        'liveness',
        'valence',
        'tempo'
    ]
    mask = (songs['metagenre'] != 'unknown')
    X = songs[features].loc[mask].values
    y = songs['metagenre'].loc[mask].values

    # remove genres
    mask = np.isin(y, genres_to_keep.index)
    X = X[mask]
    y = y[mask]

    # subsample
    genre_weights = 1 / len(genres_to_keep) /  (genres_to_keep / genres_to_keep.sum())
    probas = genre_weights[y]
    probas /= probas.sum()
    indices = np.random.choice(len(X), size=40000, replace=False, p=probas)
    X = np.ascontiguousarray(X[indices])
    y = np.ascontiguousarray(y[indices])


    #### TSNE
    initial_embeddings = initialization.pca(X, random_state=seed)

    affinities = affinity.Multiscale(
        X,
        perplexities=[200, 800],
        metric="cosine",
        n_jobs=8,
        random_state=seed,
        verbose=True,
    )

    tsne = TSNEEmbedding(
        initial_embeddings,
        affinities,
        negative_gradient_method="fft",
        n_jobs=8,
        verbose=True,
    )

    embeddings_1 = tsne.optimize(n_iter=150, exaggeration=10, momentum=0.4)
    embeddings_2 = tsne.optimize(n_iter=500, momentum=0.8)


    #### Plot
    indices = np.random.choice(len(embeddings_2), size=20000, replace=False)
    data = pd.DataFrame({
        'x': embeddings_2[indices,0],
        'y': embeddings_2[indices,1],
        'Genre': y[indices]
    })
    g = sns.relplot(x='x', y='y', hue='Genre', data=data, height=8, alpha=0.55, s=8, palette='Set1')
    g.set(xticklabels=[], xlabel=None)
    g.set(yticklabels=[], ylabel=None)
    sns.despine(bottom=True, left=True)
    plt.gca().tick_params(bottom=False, left=False)
    plt.savefig('figures/tsne_genres.pdf', bbox_inches='tight')


if __name__ == '__main__':
    app.run(main)