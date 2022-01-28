from multiprocessing.sharedctypes import Value
import pandas as pd
import numpy as np
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from absl import flags, app
from tqdm import tqdm

from multiprocessing.pool import ThreadPool


FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'data/tracks_features.csv', 'the input file')
flags.DEFINE_string('out', 'artists.csv', 'the output file')
flags.DEFINE_integer('threads', 4, 'the number of download threads', lower_bound=1)


class Downloader:

    def __init__(self):
        self.api = Spotify(
            client_credentials_manager=SpotifyClientCredentials(),
            retries=7,
            status_retries=7,
            backoff_factor=1.0
        )
        self.additional_info = {}


    def download_artists(self, ids):
        assert len(ids) <= 50  # maximum of 50 ids per request (constrained from spotify)
        artists = self.api.artists(ids)['artists']

        for artist_id, artist in zip(ids, artists):
            self.additional_info[artist_id] = {
                'name': artist['name'],
                'genres': artist['genres'],
                'followers': artist['followers']['total'],
                'popularity': artist['popularity']
            }


def download(pbar, ids):
    downloader = Downloader()

    for i in range(0, len(ids), 50):
        downloader.download_artists(ids[i:i+50])
        pbar.update(50)

    return pd.DataFrame.from_dict(downloader.additional_info, orient='index')
    

def main(argv):
    # Get ids to download
    df = pd.read_csv(FLAGS.file)
    artist_ids = list(df.artist_ids.apply(lambda l_str: eval(l_str)[0]).unique())  # check your data first ;)

    ids = np.array_split(artist_ids, FLAGS.threads)

    if 'genres' not in df:
        df = pd.DataFrame(np.nan, columns=['genres'], index=artist_ids)

    n_tracks = np.sum(df.genres.isna())

    print(f'downloading {n_tracks} artists')

    with tqdm(total=n_tracks, unit='artists', smoothing=0.0) as pbar:
        with ThreadPool(FLAGS.threads) as pool:
            download_lambda = lambda ids: download(pbar, list(ids))
            additional_infos = pool.map(download_lambda, ids)

    additional_infos = pd.concat(additional_infos)
    for column in additional_infos:
        if column not in df:
            df[column] = np.nan

    df.update(additional_infos, overwrite=True)
    
    df.to_csv(FLAGS.out)


if __name__ == '__main__':
    app.run(main)