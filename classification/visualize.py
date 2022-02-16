#!/usr/bin/env python3

"""
Visualize trained random forest classifier
"""

import pickle

import numpy as np
import sklearn.ensemble
import sklearn.model_selection
import sklearn.tree


def visualize():

    with open("model.pickle", "rb") as ifp:
        var = pickle.load(ifp)
    model = var["model"]

    print("Dicision trees:")
    for estimator in model.estimators_:
        print()
        print(sklearn.tree.export_text(estimator))


if __name__ == "__main__":
    visualize()


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
