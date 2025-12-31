"""
Machine Learning - k Nearest Neighbors
Handwritten Number Recognition
12/30/25

If you use my code or a part of it for your projects, thank you for quoting me!
"""


from sklearn.datasets import fetch_openml
from PIL import Image as Img
import numpy as np
import sys

import recreate_dataset
import performances as perf
import constants as cst
import print_results


recreate_dataset.recreateDataset()     # Change this function in 'recreate_dataset.py' or delete it if you want to use other images dataset ; Comment it if you already have images in your dataset


## Dataset ##
x = np.zeros((cst.NB_DATA_X, cst.WIDTH * cst.HEIGHT), dtype = int)
dict_x = {}
with open("dataset\\dataset_true_values.txt", "r", encoding = "utf-8") as f:
    list_correspondences = f.read().split("\n")[:-1]     # Remove the last element which the empty string
for i in range(0, cst.NB_DATA_X):
    img = Img.open(f"dataset\\img_{i}.jpg")
    x[i] = np.array(img, dtype = int).reshape(1, cst.WIDTH * cst.HEIGHT)     # 28 * 28 = 784 attributes of x[i]
    dict_x[i] = int(list_correspondences[i])     # True value of x[i]

## Training Dataset ##
with open("training_dataset\\dataset_training_index.txt", "r", encoding = "utf-8") as f:
    sample_a = f.read().split("\n")[:-1]     # Remove the last element which the empty string
a = np.zeros((cst.NB_DATA_A, cst.WIDTH * cst.HEIGHT), dtype = int)
dict_a = {}
for i, j in enumerate(sample_a):
    a[i] = x[int(j)]
    dict_a[i] = dict_x[int(j)]


if __name__ == "__main__":
    k = -1
    while k < 1 or k > cst.NB_DATA_X:
        k = int(input(f"Choose the k between 1 and {cst.NB_DATA_X}: "))
    confusion_matrix, normalized_matrix, compute_time, acc, mcc_result, list_prev, list_preci, list_recall, list_f1_score, list_fbeta_score = perf.performances(x, dict_x, sample_a, a, dict_a, k)
    print_results.printResultsOnTxt(confusion_matrix, normalized_matrix, compute_time, acc, mcc_result, list_prev, list_preci, list_recall, list_f1_score, list_fbeta_score)

    awnser = input("Do you want to determine the k optimal? (y/n) ")
    if awnser == "y":
        nb_loop = int(input("How many loops do you want for a k? (y/n) "))
        optimal_k, optimal_accuracy, compute_total_time = perf.kOptimal(x, dict_x, sample_a, a, dict_a, nb_loop, cst.NB_DATA_X)
        with open("optimal_results.txt", "w", encoding = "utf-8") as f:
            f.write(f"Optimal k: {optimal_k}\n")
            f.write(f"Optimal accuracy: {optimal_accuracy}\n")
            f.write(f"Compute total time: {compute_total_time}\n")
    else:

        sys.exit()
