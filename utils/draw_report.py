import os

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from config_classification import PHOTO_PATH


def draw_report(title, y_test, y_pred, file_name, extension):
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot()
    plt.title(f"{title}")
    if os.path.exists(PHOTO_PATH):
        plt.savefig(f"{PHOTO_PATH}/{file_name}.{extension}", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"../{PHOTO_PATH}/{file_name}.{extension}", dpi=300, bbox_inches='tight')
    plt.show()
