#!/usr/bin/env python3

"""
Visualize trained random forest regressor.
"""

import argparse
import pickle

import numpy as np
import sklearn.ensemble
import sklearn.model_selection
import sklearn.tree


def parse_args():
    """
    Parse command line arguments.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("-m", "--model", metavar="PATH", type=str, default="model.pickle",
                        help="ranfom forest model file (.pkl)")
    return parser.parse_args()


def visualize(args):
    """
    Load a random forest model and visualize it's trees.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    with open(args.model, "rb") as ifp:
        model = pickle.load(ifp)

    for index, estimator in enumerate(model.estimators_):
        print()
        print("Dicision trees [%d]" % index)
        print(sklearn.tree.export_text(estimator))
        exit()


if __name__ == "__main__":
    visualize(parse_args())


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
