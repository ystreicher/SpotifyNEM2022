import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from absl import flags, app
from pprint import pprint

from sklearn.cluster import AgglomerativeClustering

from nem.cleanup import filter_artists
from nem.util import load_artists


def cluster(artists, minimum_artists, distance_threshold):
    unique_genres = {}
    for artist_genres in artists.genres:
        for genre in artist_genres:
            unique_genres[genre] = unique_genres.get(genre, 0) + 1

    unique_genres = pd.Series(unique_genres, name='song count')
    n_genres = len(unique_genres)
    print(f'There are {n_genres} unique genres')

    # Filter genres
    unique_genres = unique_genres.loc[unique_genres > minimum_artists]
    n_genres = len(unique_genres)
    unique_genres.describe()

    genre_to_idx = {genre:i for i,genre in enumerate(unique_genres.index)}
    idx_to_genre = {i:genre for i,genre in enumerate(unique_genres.index)}     

    # build adjacency and distance matrix
    adjacency_matrix = np.zeros((n_genres, n_genres))
    for artist_genres in artists.genres:
        for genre1 in artist_genres:
            for genre2 in artist_genres:
                #if genre1 == genre2:
                #    continue
                try:
                    idx1 = genre_to_idx[genre1]
                    idx2 = genre_to_idx[genre2]
                    adjacency_matrix[idx1, idx2] += 1
                except KeyError:
                    pass

    normalized_adjacency  = adjacency_matrix / adjacency_matrix.sum(axis=1, keepdims=True)
    normalized_adjacency = (normalized_adjacency + normalized_adjacency.T) / 2
    distance_matrix =  1 / (normalized_adjacency+1)**1.0

    # Cluster
    clustering = AgglomerativeClustering(n_clusters = None, affinity='precomputed', linkage='average', distance_threshold=distance_threshold)
    labels = clustering.fit_predict(distance_matrix)

    clusters = pd.Series({idx_to_genre[i]: label for i, label in enumerate(labels)})
    return {cluster: list(genres) for cluster, genres in clusters.groupby(clusters).groups.items()}


def main(vargs):
    artists = load_artists('artists.csv')
    artists = filter_artists(artists)

    print('########### COARSE ###########')
    pprint(cluster(artists, 100, 0.995))
    print()

    print('###########  FINE  ###########')
    pprint(cluster(artists, 25, 0.997))


if __name__ == '__main__':
    app.run(main)