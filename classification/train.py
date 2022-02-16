#!/usr/bin/env python3

"""
Train ranfom forest classifier on Boston dataset
"""

import argparse
import os
import pickle

import numpy as np
import cv2 as cv
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


def load_dataset():
    """
    Load MNIST dataset and split it to train/test dataset.
    Difficulity of the dataset highly depends on the randomness of the splitting.

    Returns:
        X_train (np.ndarray): Training data of shape (n_train_samples, 256).
        X_test  (np.ndarray): Test data of shape (n_test_samples, 256).
        y_train (np.ndarray): Training label of shape (n_train_samples, ).
        y_test  (np.ndarray): Test label of shape (n_test_samples, ).
    """
    # Load train/test image data.
    def vectorise_MNIST_images(filepath, threshold=10.0):

        def convert(img):
            img = cv.resize(img, (16, 16))
            img = img > threshold
            return img.reshape((-1, )).astype(np.int8)

        # Load dataset and normalize.
        Xs = np.load(filepath)
        Xs = np.array([convert(Xs[n, :, :]) for n in range(Xs.shape[0])])

        return Xs

    # Load train/test label data.
    def vectorise_MNIST_labels(filepath):
    
        return np.load(filepath)

    if not os.path.exists("dataset/MNIST_train_images.npy"):
        raise RuntimeError("Please download MNIST data at first")

    # Load training data.
    X_train = vectorise_MNIST_images("dataset/MNIST_train_images.npy")
    y_train = vectorise_MNIST_labels("dataset/MNIST_train_labels.npy")

    # Load test data.
    X_test = vectorise_MNIST_images("dataset/MNIST_test_images.npy")
    y_test = vectorise_MNIST_labels("dataset/MNIST_test_labels.npy")

    return (X_train, X_test, y_train, y_test)


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
    X_train, X_test, y_train, y_test = load_dataset()

    # Start training for a binary random forest.
    termcolor.cprint("Train & test a binary random forest model", "yellow")

    # Binarize input.
    termcolor.cprint("  - input data shape: " + termcolor.colored(X_train.shape, "magenta"))

    # Train on the quantized data.
    model = sklearn.ensemble.RandomForestClassifier(n_estimators=args.estimators, max_depth=args.depth)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    termcolor.cprint("  - Accuracy: " + termcolor.colored(score, "magenta"))
    termcolor.cprint("")

    # Save the trained binary random forest model to a pickle file.
    termcolor.cprint("Save to a pickle", "yellow")
    with open(args.output, "wb") as ofp:
        pickle.dump({"model": model, "n_features": X_train.shape[1], "n_classes": 10}, ofp)
    termcolor.cprint("  - model: " + termcolor.colored(args.output, "magenta"))

    # Dump binarized test data.
    with open(args.dump, "wt") as ofp:
        for n in range(X_test.shape[0]):
            xs = X_test[n, :].astype(int)
            ofp.write("".join(map(str, xs.tolist())) + "\n")
    termcolor.cprint("  - test data: " + termcolor.colored(args.dump, "magenta"))

    # Run prediction for the 1st test data.
    termcolor.cprint("Run prediction for the 1st test data", "yellow")
    x_test = X_test[:1, :]
    y_pred = model.predict(x_test)[0]
    termcolor.cprint("  - y_pred: " + termcolor.colored(str(y_pred), "magenta"))


if __name__ == "__main__":
    train(parse_args())


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
