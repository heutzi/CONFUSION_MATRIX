# Confusion Matrix Generator

## Overview
This Python script is used to generate a confusion matrix for evaluating the performance of a classification model.

### Required Libraries
Install the following dependencies using pip:
```bash
pip install ast matplotlib numpy pandas scikit-learn
```

## Usage
1. Import the required libraries:
```python
import ast
import pandas as pd
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
```

2. Import data (check examples for formating):
```python
DIM = "utilité"
folder_name = "model-name"

threshold_classification = 2/6-0.01
threshold_evaluation = 2/6-0.01

predictions = build_data(folder_name, DIM)
```

3. Compute and display the confusion matrix:
```python
label_gold = np.array(list(predictions[DIM]))
label_pred = np.array(list(predictions["prediction"]))

confusion_matrix_plot(label_gold,
                      label_pred,
                      # "savename")
                      )
```

## Example Output

![Example of a confusion matrix](https://github.com/heutzi/CONFUSION_MATRIX/blob/master/results/cm-example.svg?raw=true)

## License
This project is licensed under the MIT License.

## Author
Jonas Noblet
