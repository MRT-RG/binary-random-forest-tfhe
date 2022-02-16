#!/usr/bin/env python3

"""
CuFHE code generator for binary random forest regressor
"""

import argparse
import pickle

import numpy as np
import cv2 as cv
import sklearn.ensemble


def parse_args():
    """
    Parse command line arguments.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    fmtter = lambda prog: argparse.HelpFormatter(prog,max_help_position=50)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmtter, add_help=False)

    group1 = parser.add_argument_group("Optional arguments")
    group1.add_argument("-i", "--input", metavar="PATH", type=str, default="model.pickle",
                        help="input model file (.pickle)")
    group1.add_argument("-o", "--output", metavar="PATH", type=str, default="model.cu",
                        help="output file")
    group1.add_argument("-h", "--help", action="help",
                        help="show this message and exit")

    return parser.parse_args()


class Node:
    """
    A class for representing a node in a decition tree.
    """
    def __init__(self, index, depth=None, feature=None, value=None):
        self.index = index
        self.depth = depth
        self.value = value
        self.feature = feature


def parse_tree_nodes(tree):
    """
    Parse a scikit-learn's decition tree and returns dictionary of nodes.
    """
    nodes = dict()

    # Starts with the root node id 0, depth 0, and index 1.
    stack = [(0, 0, 1)]

    while len(stack) > 0:

        # Get new node.
        # The function `pop` ensures each node is only visited once.
        node_id, depth, index = stack.pop(0)

        node = Node(index, depth, tree.feature[node_id], None)

        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if tree.children_left[node_id] != tree.children_right[node_id]:
            stack.append((tree.children_left[node_id],  depth + 1, 2 * index))
            stack.append((tree.children_right[node_id], depth + 1, 2 * index + 1))

        # If not a split node, decision result si stored in the `.value` attribute
        # of this node. In the case of regression, regression result (real value)
        # is directly stored in the `tree.value[node_id, 0, 0].
        else:
            node.value = tree.value[node_id, 0, 0]

        if node.index in nodes:
            raise RuntimeError("multiple indexes")

        nodes[node.index] = node

    return nodes


def generate_prediction_code(estimators):
    """
    Returns a string of compilable C-function for prediction of random forest on t-FHE scheme.
    """
    def generate_prediction_code_tree(nodes, tree_index, max_depth):
        """
        Returns a C-compilable string for one decision tree.
        """
        code  = "static void\n"
        code += "prediction%02d(Ctxt *c, Ctxt *d, Ctxt **o, Ctxt *c_const) {\n" % (tree_index,)
        code += "    // Prediction of the decision tree %d.\n" % (tree_index,)
        code += "\n"
        code += "    // Depth = 0\n"
        code += "    Copy(o[%d][1], c_const[1], stream[0]);\n" % (tree_index,)
        code += "    Synchronize();\n"

        index_stream = 1

        for depth in range(1, max_depth + 1):

            code += "\n    // Depth = %d\n" % (depth,)

            for index in [i for i in range(2**depth, 2**(depth + 1)) if i in nodes]:
                args = {
                    "dsrc": "d" if index % 2 == 0 else "c", # Source of data ("c" or "d").
                    "tidx": tree_index,                     # Index of decision tree.
                    "nidx": index,                          # Index of tree node.
                    "fidx": nodes[index//2].feature,        # Index of feature.
                    "pidx": index//2,                       # Index of parent node.
                    "gpu" : index_stream,                   # Index of GPU stream.
                }
                temp  = "    And(o[{tidx}][{nidx}], {dsrc}[{fidx}], o[{tidx}][{pidx}], stream[{gpu} % n_stream]);\n"
                code += temp.format(**args)

                # Increase GPU stream index.
                index_stream += 1

            code += "    Synchronize();\n"

        return code + "}\n\n"

    # Add prediction functions for each decition tree.
    code = ""
    for index, estimator in enumerate(estimators):
        nodes = parse_tree_nodes(estimator.tree_)
        depth = max(node.depth for node in nodes.values())
        code += generate_prediction_code_tree(nodes, index, depth)

    # Add main prediction function.
    code += "void\n"
    code += "prediction(Ctxt *c, Ctxt *c_const, Ctxt **o) {\n"
    code += "    // Run binary random forest on T-FHE scheme.\n"
    code += "\n"
    code += "    // Create nagation of encryptin data array, because the nagation array is\n"
    code += "    // quite useful for the prediction.\n"
    code += "    Ctxt *d = new Ctxt[n_input];\n"
    code += "    for (int n = 0; n < n_input; ++n)\n"
    code += "        Not(d[n], c[n], stream[n % n_stream]);\n"
    code += "    Synchronize();\n"
    code += "\n"
    for index in range(len(estimators)):
        code += "    prediction%02d(c, d, o, c_const);\n" % index
    code += "\n"
    code += "    delete[] d;\n"
    code += "}\n"

    return code


def generate_get_result_code(estimators):
    """
    Returns a string of compilable C-function for getting result of random forest on t-FHE scheme.
    """
    def generate_get_result_code_tree(nodes, tree_index, max_depth):
        """
        Returns a C-compilable string for one decision tree.
        """
        code  = "static double\n"
        code += "get_result%02d(Ctxt **o, Ptxt *r, PriKey& pri_key){\n" % (tree_index,)
        code += "    // Get result of the prediction of the tree %d.\n" % (tree_index,)
        code += "\n"
        code += "    double result = 0.0;\n"
        code += "\n"
        code += "    // Decrypt prediction results.\n"

        for index in range(2**(max_depth + 1)):
            if (index in nodes) and (nodes[index].value is not None):
                args = {
                    "tidx": tree_index,
                    "nidx": index,
                }
                temp  = "    Decrypt(r[{nidx}], o[{tidx}][{nidx}], pri_key);\n"
                code += temp.format(**args)

        code += "\n"
        code += "    // Compute result.\n"

        for index in range(2**(max_depth + 1)):
            if (index in nodes) and (nodes[index].value is not None):
                args = {
                    "nidx" : index,
                    "value": nodes[index].value,
                }
                temp  = "    result += r[{nidx}].message_ * {value};\n"
                code += temp.format(**args)

        code += "\n"
        code += "    return result;\n"

        return code + "}\n\n"

    # Add getting result functions for each decition tree.
    code = ""
    for index, estimator in enumerate(estimators):
        nodes = parse_tree_nodes(estimator.tree_)
        depth = max(node.depth for node in nodes.values())
        code += generate_get_result_code_tree(nodes, index, depth)

    code += "\n"
    code += "double\n"
    code += "get_result(Ctxt **o, PriKey &pri_key){\n"
    code += "    // Compute result of the prediction.\n"
    code += "\n"
    code += "    double result = 0.0;\n"
    code += "    Ptxt *r = new Ptxt[n_nodes];\n"
    code += "\n"
    for index in range(len(estimators)):
        code += "    result += get_result%02d(o, r, pri_key);\n" % index
    code += "\n"
    code += "    delete[] r;\n"
    code += "\n"
    code += "    return result / n_trees;\n"
    code += "}\n"
 
    return code


def generate_code(args):

    with open(args.input, "rb") as ifp:
        var = pickle.load(ifp)
    model = var["model"]
    n_input = var["n_features"]

    # Parse the first tree for determining tree depth.
    max_depth = 0
    for estimator in model.estimators_:
        nodes = parse_tree_nodes(estimator.tree_)
        depth = max(node.depth for node in nodes.values())
        max_depth = max(max_depth, depth)

    # Add header.
    code  = "#include <include/cufhe_gpu.cuh>\n"
    code += "using namespace cufhe;\n"
    code += "\n"
    code += "extern uint32_t n_stream;\n"
    code += "extern Stream*  stream;\n"
    code += "\n"
    code += "int n_trees = %d;\n" % len(model.estimators_)
    code += "int n_nodes = %d;\n" % (2**(max_depth + 1))
    code += "int n_input = %d;\n" % n_input
    code += "\n"

    # Add code for prediction and getting result.
    code += generate_prediction_code(model.estimators_)
    code += generate_get_result_code(model.estimators_)

    with open(args.output, "wt") as ofp:
        ofp.write(code)


if __name__ == "__main__":
    generate_code(parse_args())


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
