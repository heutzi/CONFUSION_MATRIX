import ast
import pandas as pd
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from unidecode import unidecode

import modules.quantizing as quant

# fm.fontManager.addfont("modules/cmunrm.ttf")
fm.fontManager.addfont("modules/cmunss.ttf")
# plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['font.family'] = 'CMU Sans Serif'
plt.rcParams['svg.fonttype'] = 'none'

# font = {'family' : 'CMU Serif',
#         'size'   : 11}
font = {'family': 'CMU Sans Serif',
        'size': 11}

matplotlib.rc('font', **font)


def parse(str_list: str) -> pd.DataFrame:
    """Parses csv files into pandas Dataframe

    Args:
        str_list (str): csv file

    Returns:
        pd.DataFrame: output Dataframe
    """
    str_list = ' '.join(str_list.split())
    str_list = str_list.replace(' ', ', ')
    str_list = ast.literal_eval(str_list)
    return str_list


def build_data(model_name: str, dim: str):
    predictions = pd.read_csv(
        f"predictions/{model_name}/{dim}/predictions.csv",
        sep='\t'
        )
    predictions[dim] = predictions[dim].map(
        lambda v: parse(v)
    )
    predictions["prediction"] = predictions["prediction"].map(
        lambda v: parse(v)
    )
    return predictions


def confusion_matrix_plot(labels_true: np.ndarray,
                          labels_pred: np.ndarray,
                          save_name: str = None,
                          normalize: str = 'true'):
    """Function that generates (and exports) a confusion matrix

    Args:
        labels_true (np.ndarray): True labels.
        labels_pred (np.ndarray): Predicted labels.
        save_name (str, optional):
            Export name (if None, the figure is not saved).
            Defaults to None.
        normalize (str, optional):
            Normalization, either 'true' along true labels or 'pred'
                along predicted labels.
            Defaults to 'true'.
    """
    rat = 6
    labels = quant.to_label(labels_true, rat)
    predicts = quant.to_label(labels_pred, rat)

    label_names = [f'– {int((i+1)/rat * 100)}-{int(i/rat * 100)}'
                   for i in reversed(range(rat))] + \
                  [f'+ {int(i/rat * 100)}-{int((i+1)/rat * 100)}'
                   for i in range(rat)]

    cm = confusion_matrix(labels,
                          predicts,
                          labels=range(len(label_names)),
                          normalize=normalize)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        )

    _, ax = plt.subplots(figsize=(4, 4))
    cmap = 'Blues_r'
    if normalize == 'pred':
        cmap = 'Blues'
    disp.plot(cmap=cmap,
              include_values=True,
              values_format='.2f',
              ax=ax,
              colorbar=False,
              )
    children = []

    # Prints labels only if significant (>0.2)
    for child in disp.ax_._children:
        if not type(child) == matplotlib.text.Text:
            children.append(child)
        else:
            child._fontproperties._size = 9
            if float(child._text) >= 0.2:
                child._text = f"{float(child._text):.2f}"
                children.append(child)
    disp.ax_._children = children

    label_names = [f'– {int((i+1)/rat * 100)}'
                   for i in reversed(range(rat))] + \
                  [0] +\
                  [f'+ {int((i+1)/rat * 100)}'
                   for i in range(rat)]

    ax.set_xticks(np.arange(len(label_names)) - 0.5,
                  labels=label_names)
    ax.set_xlabel('Étiquette prédite (en %)')

    ax.set_yticks(np.arange(len(label_names)) - 0.5,
                  labels=label_names)
    ax.set_ylabel('Étiquette réelle (en %)')
    ax.tick_params(axis='x', labelrotation=90)

    if save_name:
        plt.savefig(f"results/{save_name}.svg", bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_all(model_names, dims, normalize='true'):
    ext = ''
    if normalize == 'pred':
        ext = "-rec"
    for model_name in model_names:

        os.makedirs(f"results/{model_name}/", exist_ok=True)

        for dim in dims:
            all_grouped_by_id = build_data(model_name, dim)

            label_gold = np.array(list(all_grouped_by_id[dim]))
            label_pred = np.array(list(all_grouped_by_id["prediction"]))

            confusion_matrix_plot(
                label_gold, label_pred,
                unidecode(
                    f"{model_name}/Confusion{ext}-{model_name}-{dim}"
                    ),
                normalize=normalize
                )
