import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, ConfusionMatrixDisplay


def plot_confusion_matrix(
    pred, labels_max_str, traintest="TRAIN", out_path=os.path.join("outputs", "mode_choice_model")
):
    print("----- ", traintest, "results")
    print("Acc:", accuracy_score(pred, labels_max_str))
    print("Balanced Acc:", balanced_accuracy_score(pred, labels_max_str))
    for confusion_mode in [None, "true", "pred"]:
        name = "" if confusion_mode is None else confusion_mode
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ConfusionMatrixDisplay.from_predictions(
            labels_max_str, pred, normalize=confusion_mode, xticks_rotation="vertical", values_format="2.2f", ax=ax
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"random_forest_{traintest}_confusion_{name}.png"))


def mode_share_plot(labels_mobis, labels_sim, out_path=os.path.join("outputs", "mode_choice_model")):
    def mode_share_dict(labels, name):
        uni, counts = np.unique(labels, return_counts=True)
        mobis_modeshare = {u: [(c / np.sum(counts))] for u, c in zip(uni, counts)}
        df = pd.DataFrame(mobis_modeshare).swapaxes(1, 0).rename(columns={0: "Mode ratio"})
        df["Type"] = name
        df.index.name = "Mode"
        return df.reset_index()

    df_labels = pd.concat((mode_share_dict(labels_mobis, "MOBIS"), mode_share_dict(labels_sim, "Simulated")))
    plt.figure(figsize=(10, 7))
    sns.barplot(x="Mode", y="Mode ratio", hue="Type", data=df_labels)
    plt.savefig(os.path.join(out_path, "mode_share_comparison.png"))
