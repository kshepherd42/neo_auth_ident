"""
Command line script to run all of tfidf files needed to get cross validation results.

Written By: Katherine Shepherd
    ***CustomFormatter written by Dr. Sprague
"""

from os.path import exists
from selectinatortfidfs import Select
from resultsinator import Results
import numpy as np
import argparse

def set_up():
    if not exists('cpp_train.h5') or not exists('cpp_test.h5'):
        raise Exception('this needs to be run in the same folder as cpp_train and cpp_test')

    if not exists('sandbox.h5'):
        import sandboxMaker

    if not exists('tfidfs.npz'):
        import extractinatortfidfs

def run(should_select, features, files):
    set_up()
    if should_select:
        Select(features)

    return Results(files)

def printing(results, tree, avg):
    if tree:
        if avg:
            print(np.average(results.tree()))
        else:
            print(results.tree())
    else:
        if avg:
            print(np.average(results.knn()))
        else:
            print(results.knn())

'==========================================================================================='

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    """ Trick to allow both defaults and nice formatting in the help. """
    pass

def main():
    description = 'Create sandbox.h5 and extract if needed. Select features if ' \
                  'command-line arguments require you to, then print cross validation results.'

    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=description)

    parser.add_argument('--features', default = 0, help = 'number of features to be selected. if not '
                                                          'set, the number of features is not updated '
                                                          'so long as chosen.npy exists. if it does not '
                                                          'features is set to 2000.')
    parser.add_argument('--files', default = 10, help='number of files required per author.')
    parser.add_argument('--tree', default = "True", help = 'whether classifier should be Random Forest '
                                                         'Classifier or KNN.')
    parser.add_argument('--average', default = "True", help = 'whether average should be printed or the full results.')

    args = parser.parse_args()

    features = int(args.features)
    files = int(args.files)
    avg = True
    should_select = True
    tree = True

    if args.tree == "False" or args.tree == "false" or args.tree == "F" or args.tree == "f":
        tree = False

    if args.average == "False" or args.average == "false" or args.average == "F" or args.average == "f":
        avg = False

    if features == 0:
        if exists('chosen.npy'):
            should_select = False
        else:
            features = 2000


    results = run(should_select, features, files)
    printing(results, tree, avg)

if __name__ == "__main__":
    main()