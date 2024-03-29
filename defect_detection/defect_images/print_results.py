import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import argparse


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def print_results(dataset_path):
        output_folder = os.path.join(dataset_path, "output")
        results_location = os.path.join(output_folder, "output.txt")
        target_names = ['multi_pick', 'nominal', 'package_defect']

        with open(results_location, 'r') as infile:
            results = json.load(infile)
            predictions = results['predictions']
            targets = results['targets']

            y_pred = [np.argmax(softmax(p)) for p in predictions]
            cnf_matrix = confusion_matrix(targets, y_pred)
            FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
            FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            TP = np.diag(cnf_matrix)
            TN = cnf_matrix.sum() - (FP + FN + TP)

            FP = FP.astype(float)
            FN = FN.astype(float)
            TP = TP.astype(float)
            TN = TN.astype(float)

            FPR = FP/(FP+TN)

            recall = TP / (TP + FN)

            for name, fpr, rec in zip(target_names, FPR, recall):
                 print (name, '\trecall = ', rec, '\tfpr = ', fpr)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", required=True)
    
    args = parser.parse_args()
    dataset_path = args.dataset_path

    print_results(dataset_path)