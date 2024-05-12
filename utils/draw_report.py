from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from config_classification import PHOTO_PATH, LAST_NAME


def draw_report(title, y_test, y_pred, file_name, extension):
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot()
    plt.title(f"{title}")
    plt.savefig(f"{PHOTO_PATH}/{file_name}{LAST_NAME}.{extension}", dpi=300, bbox_inches='tight')
    plt.show()