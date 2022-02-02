import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from absl import flags, app

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import auc, roc_curve, classification_report

from tueplots import bundles

FLAGS = flags.FLAGS
flags.DEFINE_string('track_file', 'songs_filtered.csv', 'the tracks file')

RELEVANT_FEATURES = [
    'danceability',
    'energy',
    'key',
    'loudness',
    'mode',
    'speechiness',
    'acousticness',
    'instrumentalness',  # extremly discriminative!
    'liveness',
    'valence',
    'tempo',
    'duration_ms',
    'popularity',
]


def train_and_evaluate_logistic(X, y, cv_folds = 5, train_size=0.8):

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
    
    ## ML Pipeline
    pipe = make_pipeline(
        StandardScaler(), 
        GridSearchCV(
            LogisticRegression(),
            param_grid={
            'penalty': ['l2'],
            'class_weight': ['balanced'],
            # 'l1_ratio': [0.5],
            # 'solver': ['saga'],
            'C': np.logspace(-3, 3, 5)
            },
            cv=cv_folds,
            refit=True
        )
    )

    # train
    pipe.fit(X_train, y_train)

    # evaluate (i) ROC and AUC
    # calculate the fpr and tpr for all thresholds of the classification
    probs = pipe.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    # evaluate (ii) metrics for logistic regression
    predictions = pipe.predict(X_test)
    report = classification_report(y_test, predictions)

    # evaluate (iii) calibration curve
    y_prob = pipe.predict_proba(X_test)[:, 1]
    (prob_true, prob_pred) = calibration_curve(y_test, y_prob, n_bins=5, strategy="uniform")
    

    # bit of ugly...
    coefficients = np.ravel(pipe.steps[1][1].best_estimator_.coef_)

    return (fpr, tpr, roc_auc), report, coefficients, (prob_true, prob_pred, y_prob)

def plot_coefs_barth(coefficients, model_name, filename):
    plt.rcParams.update(bundles.neurips2021())
    fig = plt.figure(figsize=(2.7, 2.7))
    plt.clf()
    coefs = pd.DataFrame(
        np.ravel(coefficients),
        columns=["Coefficients"],
        index=['danceability',
            'energy',
            'key',
            'loudness',
            'mode',
            'speechiness',
            'acousticness',
            'instrumentalness',  # extremly discriminative!
            'liveness',
            'valence',
            'tempo',
            'duration ms', # must write this here, as underscore breaks the plot
            ],
    )
    coefs.plot(kind="barh", figsize=(2.7, 2.7))
    plt.title(model_name)
    plt.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.3)

    plt.savefig(f'figures/logistic_coefs_{filename}.pdf', bbox_inches='tight')

def plot_calibration_curve(calibration_plots, filename):
    plt.rcParams.update(bundles.neurips2021())
    fig = plt.figure(figsize=(2.7, 2.7))
    plt.clf()
    plt.plot([0, 1], [0, 1],'r--')

    for cali_plot in calibration_plots:
        (prob_true, prob_pred, y_prob), name = cali_plot
        plt.plot(prob_pred, prob_true, label=name, marker='o')

    plt.xlim([-.1, 1.1])
    plt.ylim([-.1, 1.1])
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration plot")
    plt.legend()
    plt.savefig(f'figures/calibration_{filename}.pdf', bbox_inches='tight')

def plot_auc_combined(auc1, auc2, names=["first", "second"]):
    plt.rcParams.update(bundles.neurips2021())
    fig = plt.figure(figsize=(2.7, 2.7))
    plt.clf()
    (fpr1, tpr1, roc_auc1) = auc1
    (fpr2, tpr2, roc_auc2) = auc2

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr1, tpr1, 'b', label = 'AUC = {:.2f}, {}'.format(roc_auc1, names[0]))
    plt.plot(fpr2, tpr2, 'g', label = 'AUC = {:.2f}, {}'.format(roc_auc2, names[1]))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f'figures/roc_logistic.pdf', bbox_inches='tight')

def get_tracks_and_labels() -> pd.DataFrame:
    tracks = pd.read_csv(FLAGS.track_file)
    tracks_relevant = tracks[RELEVANT_FEATURES]

    n_features = tracks_relevant.values.shape[1] - 1

    X = tracks_relevant.values[:, :n_features]
    y = tracks_relevant.values[:, n_features]

    return (X, y)

def main(argv):
    tracks = pd.read_csv(FLAGS.track_file)

    tracks_relevant = tracks[RELEVANT_FEATURES]

    n_features = tracks_relevant.values.shape[1] - 1

    X = tracks_relevant.values[:, :n_features]
    y = tracks_relevant.values[:, n_features]

    y_labels_general =  np.where(y > np.quantile(y, 0.5), 1, 0)
    y_labels_tight =  np.where(y > np.quantile(y, 0.9), 1, 0)

    auc_plot_general, evaluation_general, coeffs_general, calibration_plot_general = train_and_evaluate_logistic(X, y_labels_general)
    auc_plot_tight, evaluation_tight, coeffs_tight, calibration_plot_tight = train_and_evaluate_logistic(X, y_labels_tight)

    ## plot and save! 
    plot_auc_combined(auc_plot_general, auc_plot_tight, names=["50\% threshold model", "10\% threshold model"])

    plot_coefs_barth(coeffs_general, model_name="50\% threshold model", filename="50_threshold_model")
    plot_coefs_barth(coeffs_tight, model_name="weights 10\% threshold model", filename="10_threshold_model")

    plot_calibration_curve(
        (
            (calibration_plot_general, "50\% threshold"),
            (calibration_plot_tight, "10\% threshold")
        ) , filename="combined")

    ## show metrics in terminal
    print("50% threshold model: ")
    print(evaluation_general)
    print()
    print("10% threshold model")
    print(evaluation_tight)




if __name__ == '__main__':
    app.run(main)
