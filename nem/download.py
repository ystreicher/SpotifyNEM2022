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
flags.DEFINE_string('out', 'augmented_ds.csv', 'the output file')
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


    def download_tracks(self, ids):
        assert len(ids) <= 50

        try:
            tracks = self.api.tracks(ids)['tracks']
        except Exception as e:
            print(f'exc during track download: {repr(e)}')
            return
        
        for track_id, track in zip(ids, tracks):
            if track is None:
                print(f'invalid track id: {track_id}')
                continue

            if track_id not in self.additional_info:
                self.additional_info[track_id] = {
                    'popularity': track['popularity'],
                    'album_type': track['album']['album_type'],
                }


def download(pbar, ids):
    downloader = Downloader()

    for i in range(0, len(ids), 50):
        downloader.download_tracks(ids[i:i+50])
        pbar.update(50)

    return pd.DataFrame.from_dict(downloader.additional_info, orient='index')
    

def main(argv):
    # Get ids to download
    df = pd.read_csv(FLAGS.file)
    df = df.set_index('id')

    if 'popularity' not in df:
        df['popularity'] = np.nan

    ids = np.array_split(df.loc[df.popularity.isna()].index.values, FLAGS.threads)
    n_tracks = np.sum(df.popularity.isna())

    print(f'downloading {n_tracks} tracks')
    with tqdm(total=n_tracks, unit='tracks', smoothing=0.0) as pbar:
        with ThreadPool(FLAGS.threads) as pool:
            download_lambda = lambda ids: download(pbar, list(ids))
            additional_infos = pool.map(download_lambda, ids)

    print('saving')

    additional_infos = pd.concat(additional_infos)
    for column in additional_infos:
        if column not in df:
            df[column] = np.nan

    df.update(additional_infos, overwrite=True)
    
    df.to_csv(FLAGS.out)


if __name__ == '__main__':
    app.run(main)
