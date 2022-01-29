import pandas as pd
import numpy as np
from absl import flags, app
from tqdm import tqdm

from multiprocessing.pool import ThreadPool


FLAGS = flags.FLAGS
flags.DEFINE_string('artist_file', 'data/artists.csv', 'the tracks file')
flags.DEFINE_string('track_file', 'data/augmented_ds.csv', 'the artists file')
flags.DEFINE_string('out_artists', 'artists_filtered.csv', 'the output file')
flags.DEFINE_string('out_combined', 'songs_artists_combined.csv', 'the output file')


def without_outliers(artists):

    # (i): remove artists without followers and with extremely many follis
    print("(i) removing 0.05 and 0.95 percentiles of log followers of artists")
    p5, p95 = artists.followers_log.quantile([0.05, 0.95])
    print(f'5% percentile: {p5}, 95% percentile: {p95}')
    artists_no_quantiles = artists.loc[(artists.followers_log > p5) & (artists.followers_log < p95)]
    
    n_perct_removed = artists.shape[0] - artists_no_quantiles.shape[0]
    print(f"{n_perct_removed} artists removed")

    # (ii) consider followers_log vs popularity, spread is large, many outliers
    ## who are those? Youtube /insta stars that are shitty musicians etc
    ## PCA to cut off horizontally
    print("(ii) doing pca to cutoff outliers of log_follis vs. popularity")
    data_mat = artists.loc[(artists.followers_log > p5) & (artists.followers_log < p95)][['followers_log', 'popularity']].values

    X = (data_mat - np.mean(data_mat, axis=0, keepdims=True)) / np.std(data_mat, axis=0, keepdims=True)
    V, Sigma, _ = np.linalg.svd(np.cov(X, rowvar=False), hermitian=True)

    projection = X @ V
    vals_first_pc = projection[:, 0]
    hor_cutoff = np.where(np.fabs(projection[:, 1]) < 0.8)

    ## results of (i) as new dataframe
    ## and finally remove those outliers!
    artists_no_quantiles["val_first_pc"] = vals_first_pc
    filtered_artists = artists_no_quantiles.iloc[hor_cutoff]

    n_pca_removed = artists_no_quantiles.shape[0] - filtered_artists.shape[0]
    print(f"pca removed: {n_pca_removed}")

    # (iii) filter out those artists without valid genre
    print("(iii): removing artists without genres")
    filtered_artists_with_genres = filtered_artists.query('`genres` != "[]"')
    first_genre_as_string = filtered_artists_with_genres['genres'].apply(
        lambda list_string: str(eval(list_string)[0])
    )
    n_without_genres = filtered_artists.shape[0] - filtered_artists_with_genres.shape[0]
    print(f"there are {n_without_genres} artists without genre")
    print("we only take those with")

    # take only the first genre for the dataset
    filtered_artists_with_genres.loc[:, ['genres']] = first_genre_as_string

    ## metrics
    total_removed = n_perct_removed + n_pca_removed + n_without_genres
    fraction_of_removed = total_removed / artists.shape[0]

    print()
    print(f"END: In total we removed {total_removed} artists")
    print(f"END: This is {fraction_of_removed} of the original ones")
    print(f"END: saving {filtered_artists_with_genres.shape[0]} artists")

    return filtered_artists_with_genres


def merge_tracks_and_artists(tracks, artists) -> pd.DataFrame:

    # unsafe!
    print("selecting tracks by our new artists")
    tracks.artist_ids = tracks['artist_ids'].apply(lambda liststring: eval(liststring)[0])
    aug_dataset = artists.merge(tracks, left_on='artist_id', right_on='artist_ids')
    n_selected_tracks = aug_dataset.shape[0]
    print(f"{n_selected_tracks} selected")

    print(f"We have the following features availalbe: {aug_dataset.columns}")

    relevant_columns = [
    # 'artist_id', 
    # 'genres', 
    # 'artist_name', 
    # 'artist_followers',       
    # 'artist_popularity', 
    # 'artist_followers_log', 
    # 'val_first_pc',
    # 'id', 
    # 'name', 
    # 'album',
    # 'album_id', 
    # 'artists', 
    # 'artist_ids', 
    'track_number', 
    # 'disc_number',
    # 'explicit', 
    'danceability', 
    'energy', 
    'key', 
    'loudness', 
    'mode',
    'speechiness', 
    'acousticness', 
    'instrumentalness', 
    'liveness',
    'valence', 
    'tempo', 
    'duration_ms', 
    'time_signature', 
    'year',
    # 'release_date', 
    'popularity', 
    # 'album_type'
    ]

    print(f"We chose {relevant_columns}")

    ds_relevant_features = aug_dataset[relevant_columns]

    print(f"saving {n_selected_tracks} tracks")
    
    return ds_relevant_features


def main(argv):

    # Get ids to download
    tracks = pd.read_csv(FLAGS.track_file)
    artists = pd.read_csv(FLAGS.artist_file)
    artists['followers_log'] = np.log10(artists.followers + 1)
    print(f"In total we have {artists.shape[0]} artists")

    artists = without_outliers(artists)

    artists.rename(columns={
        'Unnamed: 0':'artist_id',
        'name': 'artist_name',
        'followers': 'artist_followers',
        'popularity': 'artist_popularity',
        'followers_log': 'artist_followers_log',
        }, inplace=True)

    artists_tracks_combined = merge_tracks_and_artists(tracks, artists)
    
    artists.to_csv(FLAGS.out_artists, index=False)

    artists_tracks_combined.to_csv(FLAGS.out_combined, index=False)



if __name__ == '__main__':
    app.run(main)