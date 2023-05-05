import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
)
from sklearn.metrics import recall_score, balanced_accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt


def compute_preformence(T, S, Y):
    perf = {}
    perf["AUC"] = roc_auc_score(T, S)
    precision, recall, threshold = metrics.precision_recall_curve(T, S)
    perf["PR_AUC"] = metrics.auc(recall, precision)
    perf["BACC"] = balanced_accuracy_score(T, Y)
    tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
    TPR = tp / (tp + fn)
    perf["precision"] = precision_score(T, Y)
    perf["accuracy"] = accuracy_score(T, Y)
    perf["KAPPA"] = cohen_kappa_score(T, Y)
    perf["recall"] = recall_score(T, Y)
    return perf


def plot_training_epoch(train_info, val_info, file):

    # Plot train/validation loss and AUC
    epochs = range(1, len(train_info["loss"]) + 1)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].plot(epochs, train_info["loss"], label="Training Loss", color="blue")
    ax[0].plot(epochs, val_info["loss"], label="Validation Loss", color="red")
    ax[0].set_title("Training and validation Loss", fontsize=15)
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(epochs, train_info["acc"], label="Training Accuracy", color="blue")
    ax[1].plot(epochs, val_info["acc"], label="Validation Accuracy", color="red")
    ax[1].set_title("Training and validation Accuracy", fontsize=15)
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    ax[2].plot(epochs, train_info["auc"], label="Training AUC", color="blue")
    ax[2].plot(epochs, val_info["auc"], label="Validation AUC", color="red")
    ax[2].set_title("Training and validation AUC", fontsize=15)
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("AUC")
    ax[2].legend()

    if not os.path.exists("trained_model/plots"):
        os.makedirs("trained_model/plots")

    plt.savefig("trained_model/plots/" + file + ".png")
    plt.close()

