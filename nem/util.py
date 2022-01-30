import numpy as np
import pandas as pd
import ast


class ZNorm:

    def __init__(self, data):
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)

    def normalize(self, X):
        return (X - self.mean) / self.std

    
    def denormalize(self, X):
        return X * self.std + self.mean


def subsample2d(X, max_per_bin=25  , bins=51):
    hist2d, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], bins=bins)
    
    new_chunks = []
    for i in range(hist2d.shape[0]):
        for j in range(hist2d.shape[0]):
            mask = (X[:, 0] >= xedges[i]) & (X[:, 0] < xedges[i+1]) & (X[:, 1] >= yedges[j]) & (X[:, 1] < yedges[j+1])
            chunk = X[mask]
            if np.sum(mask) <= max_per_bin:
               new_chunks += [chunk]
            else:
                indices = np.random.choice(chunk.shape[0], max_per_bin, replace=False)
                new_chunks += [chunk[indices]]

    return np.concatenate(new_chunks)


def load_songs(file):
    df = pd.read_csv(file)
    
    df.artists = df.artists.apply(ast.literal_eval)
    df.artist_ids = df.artist_ids.apply(ast.literal_eval)
    df = df.set_index('id')
    
    if 'genres' in df:
        df.genres = df.genres.apply(ast.literal_eval)


    return df


def load_artists(file):
    df = pd.read_csv(file)
    
    df.genres = df.genres.apply(ast.literal_eval)
    if 'id' not in df.columns:
        df['id'] = df['Unnamed: 0']
        del df['Unnamed: 0']
    df = df.set_index('id')
    
    return df



def reverse_dict(dic):
    reverse = {}
    for key, values in dic.items():
        for v in values:
            reverse[v] = key

    return reverse