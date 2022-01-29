import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags
from tueplots import bundles


from nem.util import ZNorm, subsample2d


flags.DEFINE_string('artists', 'artists.csv', 'path to artists dataset')
flags.DEFINE_string('songs', 'augmented_ds.csv', 'path to augmented song dataset')
flags.DEFINE_string('out', 'songs_filtered.csv', 'out path')
flags.DEFINE_float('unpopular_threshold', 0.5, 'at which quantile to classify an artists as unpopular', lower_bound=0.0, upper_bound=1.0)
FLAGS = flags.FLAGS


def plot_artist_filtering(original, filtered):
    def plot(X):
        subsampled = subsample2d(X)

        plt.figure()
        plt.scatter(np.power(10, subsampled[:,0]) - 1, subsampled[:, 1], s=0.3)
        plt.xlabel('Followers')
        plt.ylabel('Popularity')
        plt.xscale('log')
        plt.xlim(10**0, 10**8)
        plt.ylim(0, 100)
    
    plot(original)
    plt.title('Unfiltered Artists')
    plt.savefig(f'figures/artists_unfiltered.pdf', bbox_inches='tight')
    plt.close()

    plot(filtered)
    plt.title('Filtered Artists')
    plt.savefig(f'figures/artists_filtered.pdf', bbox_inches='tight')
    plt.close()



def filter_artists(artists):
    data_mat = np.empty((artists.shape[0], 2))
    data_mat[:, 0] = np.log10(artists.followers + 1)
    data_mat[:, 1] = artists.popularity.values

    print('total artists: ', data_mat.shape[0])

    # SVD
    znorm = ZNorm(data_mat)
    X = znorm.normalize(data_mat)
    V, Sigma, _ = np.linalg.svd(np.cov(X, rowvar=False), hermitian=True)
    transformed = X @ V

    # outlier removal (artists far away from the first principal axis, see scatter plots in the notebooks)
    is_outlier = np.fabs(transformed[:,1]) > 0.8
    n_outliers = is_outlier.sum()
    print(f'outliers: {n_outliers} ({100 * n_outliers / data_mat.shape[0]:.2f}%)')
    
    # unpopular artists removal
    q = np.quantile(transformed[:, 0], FLAGS.unpopular_threshold)
    q_log_follower, q_popularity = znorm.denormalize([[q, 0.0]] @ V.T)[0]
    is_unpopular = transformed[:, 0] < q
    n_unpopular = is_unpopular.sum()
    print(f'unpopular: {n_unpopular} ({100 * n_unpopular / data_mat.shape[0]:.0f}%)')
    print(f'\t* apprx. less than {np.power(10, q_log_follower) - 1:.1f} follower and {q_popularity:.1f} popularity')

    to_keep = ~(is_outlier | is_unpopular)
    print(f'keeping {to_keep.sum()} artists ({100 * to_keep.sum() / data_mat.shape[0]:.2f}% of all artists)')
    filtered_artists = znorm.denormalize(transformed[to_keep] @ V.T)

    # plot filtering figures
    plot_artist_filtering(data_mat, filtered_artists)

    return artists.loc[to_keep]


def main(vargs):
    plt.rcParams.update(bundles.neurips2021())

    tracks = pd.read_csv(FLAGS.songs)
    artists = pd.read_csv(FLAGS.artists)

    filtered_artists = filter_artists(artists)


if __name__ == '__main__':
    app.run(main)