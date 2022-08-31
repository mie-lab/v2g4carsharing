import os
import matplotlib.pyplot as plt
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
