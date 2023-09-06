import time

import matplotlib
from sklearn.svm import SVC
import roll_data
kernels = ['linear', 'rbf', 'sigmoid']
kernel_names = ['SVM-' + kernel for kernel in kernels]
import numpy as np
train_data, test_data, target_data, test_target_data = roll_data.rolled_data(6)
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
for i in range(len(kernels)):
    mean_error=[]
    std_error=[]
    Ci_range = [0.01, 0.1, 1, 5, 10, 25, 50, 100]
    for Ci in Ci_range:
        clf = SVC(kernel=kernels[i], random_state=0,C=Ci)
        
        time_start = time.time()

        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(clf,train_data[features], train_data['FTR'], cv=5) #test train division will be inside the function. So using full data
        time_run = time.time() - time_start

        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        
        if(Ci==0.01):
            from sklearn.metrics import confusion_matrix
            
            conf_matrix_lg = confusion_matrix(y_test, y_pred)
            plot_2 = get_plot_fn()
            ax1 = plot_2.subplot()
            sns.heatmap(conf_matrix_lg, annot=True, fmt='g', ax=ax1);  #annot=True to annotate cells, ftm='g' to disable scientific notation
            # labels, title and ticks
            ax1.set_xlabel('Predicted labels');ax1.set_ylabel('True labels');
            ax1.set_title('Confusion Matrix'+'C = 0.01'+'Kernel: '+kernels[i]);
            ax1.xaxis.set_ticklabels(['Draw', 'Away Win', 'Home Win']); ax1.yaxis.set_ticklabels(['Draw', 'Away Win', 'Home Win']);
            plot_2.show()
            

            knn_score = clf.predict(X_test)
            from sklearn import metrics
            fpr_k_A, tpr_k_A, _ = metrics.roc_curve(y_test, knn_score, pos_label=-1)
            a = metrics.auc(fpr_k_A, tpr_k_A)
            fpr_k_D, tpr_k_D, _ = metrics.roc_curve(y_test, knn_score, pos_label=0)
            d = metrics.auc(fpr_k_D, tpr_k_D)
            fpr_k_W, tpr_k_W, _ = metrics.roc_curve(y_test, knn_score, pos_label=1)
            w = metrics.auc(fpr_k_W, fpr_k_W)

            ruc_plot = get_plot_fn()
            ruc_plot.plot(fpr_k_A, tpr_k_A, c='black')
            ruc_plot.plot(fpr_k_D, tpr_k_D, c='r')
            ruc_plot.plot(fpr_k_W, tpr_k_W, c='green')
            ruc_plot.plot([0,1],[0,1], 'b--')
            ruc_plot.title('ROC Curve For '+kernels[i])
            ruc_plot.xlabel('False positive rate')
            ruc_plot.ylabel('True positive rate')
            ruc_plot.legend([f'Class 0 (auc: {a})', f'Class 1 (auc: {w})', f'Class -1 (auc: {a})', 'Random Classifier'], bbox_to_anchor=(1.04, 1),
                            loc="upper left", fontsize='6')
            ruc_plot.show()
