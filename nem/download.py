import pandas as pd
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from absl import flags, app
from tqdm import tqdm


FLAGS = flags.FLAGS
flags.DEFINE_string('file', 'tracks_features.csv', 'the input file')
flags.DEFINE_string('out', 'tracks_features_augmented.csv', 'the output file')


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


def main(argv):
    # Get ids to download
    df = pd.read_csv(FLAGS.file)
    ids = list(df.id)

    downloader = Downloader()

    print(f'downloading {len(ids)} tracks')
    with tqdm(total=len(ids), unit='tracks', smoothing=0.005) as pbar:
        for i in range(0, len(ids), 50):
            downloader.download_tracks(ids[i:i+50])
            pbar.update(50)

    print('saving')

    df = df.set_index('id')
    popularity = pd.Series(downloader.popularity, index=downloader.ids)
    df['popularity'] = popularity
    
    df.to_csv(FLAGS.out)


if __name__ == '__main__':
    app.run(main)