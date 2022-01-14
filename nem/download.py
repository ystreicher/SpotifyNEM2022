import pandas as pd
import numpy as np
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from absl import flags, app
from tqdm import tqdm

from multiprocessing.pool import ThreadPool


FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'tracks_features.csv', 'the input file')
flags.DEFINE_string('out', 'augmented_ds.csv', 'the output file')
flags.DEFINE_integer('threads', 4, 'the number of download threads', lower_bound=1)


class Downloader:

    def __init__(self):
        self.api = Spotify(
            client_credentials_manager=SpotifyClientCredentials()
        )
        self.ids = []
        self.popularity=[]


    def download_tracks(self, ids):
        assert len(ids) <= 50
        tracks = self.api.tracks(ids)['tracks']

        self.ids += ids
        self.popularity += [track['popularity'] for track in tracks]


def download(pbar, ids):
    downloader = Downloader()

    for i in range(0, len(ids), 50):
        downloader.download_tracks(ids[i:i+50])
        pbar.update(50)

    return pd.Series(downloader.popularity, index=downloader.ids)
    

def main(argv):
    # Get ids to download
    df = pd.read_csv(FLAGS.file)
    ids = np.array_split(df.id.values, FLAGS.threads)

    print(f'downloading {len(df.id)} tracks')
    with tqdm(total=len(df.id), unit='tracks', smoothing=0.0) as pbar:
        with ThreadPool(FLAGS.threads) as pool:
            download_lambda = lambda ids: download(pbar, list(ids))
            popularity = pool.map(download_lambda, ids)

    print('saving')

    df = df.set_index('id')
    popularity = pd.concat(popularity)
    df['popularity'] = popularity
    
    df.to_csv(FLAGS.out)


if __name__ == '__main__':
    app.run(main)