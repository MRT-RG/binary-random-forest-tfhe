#!/usr/bin/env python3

"""
Train ranfom forest classifier on Boston dataset
"""

import argparse
import pickle

import numpy as np
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import termcolor


def parse_args():
    """
    Parse command line arguments.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    fmtter = lambda prog: argparse.HelpFormatter(prog,max_help_position=50)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmtter, add_help=False)

    group1 = parser.add_argument_group("Optional arguments for a random forest model")
    group1.add_argument("-b", "--bins", metavar="INT", type=int, default=10,
                        help="number of quantization bins")
    group1.add_argument("-d", "--depth", metavar="INT", type=int, default=5,
                        help="max depth")
    group1.add_argument("-e", "--estimators", metavar="INT", type=int, default=10,
                        help="number of estimators")

    group2 = parser.add_argument_group("Other optional arguments")
    group2.add_argument("-o", "--output", metavar="PATH", type=str, default="model.pickle",
                        help="output model file path")
    group2.add_argument("-u", "--dump", metavar="PATH", type=str, default="testdata.txt",
                        help="binarized test data file path")
    group2.add_argument("-s", "--seed", metavar="INT", type=int, default=None,
                        help="random seed")
    group2.add_argument("-h", "--help", action="help",
                        help="show this message and exit")

    return parser.parse_args()


def load_dataset(test_size):
    """
    Load Boston housing dataset and split it to train/test dataset.
    Difficulity of the dataset highly depends on the randomness of the splitting.

    Args:
        test_size (float): Ratio fo the test dataset.

    Returns:
        X_train (np.ndarray): Training data of shape (n_train_samples, 13).
        X_test  (np.ndarray): Test data of shape (n_test_samples, 13).
        y_train (np.ndarray): Training label of shape (n_train_samples, ).
        y_test  (np.ndarray): Test label of shape (n_test_samples, ).
    """
    data = sklearn.datasets.load_boston()
    X = data["data"]
    y = data["target"]

    return sklearn.model_selection.train_test_split(X, y, test_size=test_size)


def encode(X, n_bins=10):
    """
    Encode the given floating-number matrix to a binarized matrix.
    For the details of the binarization, see <docs/binarization.md>.

    Args:
        X      (np.ndarray): Input matrix of shape (n_samples, n_features).
        n_bins (int)       : Number of bins.

    Returns:
        Z (np.ndarray): Binarized matrix of shape (n_samples, n_bins * n_features).
    """
    # Create output matrices and initialize them as zero,
    # where Zs[k] corresponds to a binarized matrix of the feature vector X[:, k].
    Zs = [np.zeros((X.shape[0], n_bins)) for _ in range(X.shape[1])]

    # Repeat for all n-th features.
    for n in range(X.shape[1]):

        # Compute min value, max value and delta (width of bin) of the n-th feature.
        v_min = np.min(X[:, n])
        v_max = np.max(X[:, n])
        v_dlt = (v_max - v_min) / n_bins

        # Compute a matrix of indices.
        flags = np.floor((X[:, n] - v_min) / v_dlt).astype(np.int)

        # Make binarized matrix.
        for k in range(X.shape[0]):
            Zs[n][k, :flags[k]] = 1

    return np.concatenate(Zs, axis=1)


# Main procedure.
def train(args):
    """
    Train a random forest model.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    # Dump command line arguments.
    termcolor.cprint("Dump command line arguments", "yellow")
    termcolor.cprint("  - args: " + str(args))
    termcolor.cprint("")

    # Fix random seed.
    if args.seed:
        np.random.seed(args.seed)

    # Load Boston housing dataset.
    X_train, X_test, y_train, y_test = load_dataset(test_size=0.2)

    # Train and test a lnear regression model as a reference.
    termcolor.cprint("Train & test a linear regression model", "yellow")
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    termcolor.cprint("  - R2 score: " + termcolor.colored(score, "magenta"))
    termcolor.cprint("")

    # Start training for a binary random forest.
    termcolor.cprint("Train & test a binary random forest model", "yellow")

    # Binarize input.
    Z_train = encode(X_train, args.bins)
    Z_test  = encode(X_test,  args.bins)
    termcolor.cprint("  - original matrix shape: " + termcolor.colored(X_train.shape, "magenta"))
    termcolor.cprint("  - binarized input shape: " + termcolor.colored(Z_train.shape, "magenta"))

    # Train on the quantized data.
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=args.estimators, max_depth=args.depth)
    model.fit(Z_train, y_train)
    score = model.score(Z_test, y_test)
    termcolor.cprint("  - R2 score: " + termcolor.colored(score, "magenta"))
    termcolor.cprint("")

    # Save the trained binary random forest model to a pickle file.
    termcolor.cprint("Save to a pickle", "yellow")
    with open(args.output, "wb") as ofp:
        pickle.dump({"model": model, "n_features": Z_train.shape[1]}, ofp)
    termcolor.cprint("  - model: " + termcolor.colored(args.output, "magenta"))

    # Dump binarized test data.
    with open(args.dump, "wt") as ofp:
        for n in range(Z_test.shape[0]):
            zs = Z_test[n, :].astype(int)
            ofp.write("".join(map(str, zs.tolist())) + "\n")
    termcolor.cprint("  - test data: " + termcolor.colored(args.dump, "magenta"))

    # Run prediction for the 1st test data.
    termcolor.cprint("Run prediction for the 1st test data", "yellow")
    z_test = Z_test[:1, :]
    y_pred = model.predict(z_test)[0]
    termcolor.cprint("  - y_pred: " + termcolor.colored(str(y_pred), "magenta"))


if __name__ == "__main__":
    train(parse_args())


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
