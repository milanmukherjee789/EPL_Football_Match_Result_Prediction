import math
import time

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import roll_data

def get_plot_fn():
    plot = matplotlib.pyplot
    font = {'family': 'serif',
            'weight': 'light',
            'size': '6'}
    plot.rc('font', **font)
    plot.rcParams['figure.constrained_layout.use'] = True
    plot.rcParams['figure.dpi'] = 150
    plot.rc('xtick', labelsize=6)
    plot.rc('ytick', labelsize=6)
    plot.rc('axes', labelsize=6)

    return plot

rf_clf = SVC(kernel='linear', random_state=0)
train_data, test_data, target_data, test_target_data = roll_data.rolled_data(6)

rf_clf.fit(train_data, target_data)
preds = rf_clf.predict(test_data)
accuracy = accuracy_score(test_target_data, preds)
print(accuracy)
print(classification_report(test_target_data, preds))
conf_matrix_lg = confusion_matrix(test_target_data, preds)
plot_2 = get_plot_fn()
ax1 = plot_2.subplot()
sns.heatmap(conf_matrix_lg, annot=True, fmt='g', ax=ax1);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax1.set_xlabel('Predicted labels');ax1.set_ylabel('True labels');
ax1.set_title('Confusion Matrix');
ax1.xaxis.set_ticklabels(['Draw', 'Away Win', 'Home Win']); ax1.yaxis.set_ticklabels(['Draw', 'Away Win', 'Home Win']);
print(conf_matrix_lg)
conf_matrix_lg
plot_2.show()
knn_score = rf_clf.predict_proba(test_data)[:,0]
from sklearn import metrics
fpr_k_A, tpr_k_A, _ = roc_curve(test_target_data, knn_score, pos_label=-1)
a = metrics.auc(fpr_k_A, tpr_k_A)
fpr_k_D, tpr_k_D, _ = roc_curve(test_target_data, knn_score, pos_label=0)
d = metrics.auc(fpr_k_D, tpr_k_D)
fpr_k_W, tpr_k_W, _ = roc_curve(test_target_data, knn_score, pos_label=1)
w = metrics.auc(fpr_k_W, fpr_k_W)


ruc_plot = get_plot_fn()
ruc_plot.plot(fpr_k_A, tpr_k_A, c='black')
ruc_plot.plot(fpr_k_D, tpr_k_D, c='r')
ruc_plot.plot(fpr_k_W, tpr_k_W, c='green')
ruc_plot.plot([0,1],[0,1], 'b--')
ruc_plot.xlabel('False positive rate')
ruc_plot.ylabel('True positive rate')
ruc_plot.legend([f'Class 0 (auc: {a})', f'Class 1 (auc: {w})', f'Class -1 (auc: {a})', 'Random Classifier'], bbox_to_anchor=(1.04, 1),
                loc="upper left", fontsize='6')
ruc_plot.show()