import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.dummy import DummyClassifier
import time, math
from sklearn.metrics import accuracy_score
import matplotlib as matplotlib
import seaborn as sns


def rolling_averages(group, cols, new_cols, roll_match):
    group = group.sort_values("Date")
    # zscore = lambda x: (x.values[-1] - x.mean()) / x.std(ddof=1)
    # rolling_stats = (group[cols] - group[cols].rolling(roll_match, closed='left')) /group[cols].rolling(roll_match).std()
    rolling_stats = group[cols].rolling(roll_match, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


def get_team_form(group, cols, new_col, roll_match):
    group = group.sort_values('Date')
    form_match = group[cols].rolling(roll_match, closed='left').apply(lambda x: x.mode()[0])
    print(form_match)
    group[new_col] = form_match
    group = group.dropna(subset=new_col)
    return group


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


cols = ["FTHG", "FTAG", 'ref_xg_home', 'ref_xg_away', 'FTR']
new_cols = [f"{c}_avg" for c in cols]

df = pd.read_csv(f'final_df.csv')

df_roll = df.groupby('HomeTeam').apply(lambda x: rolling_averages(x, cols, new_cols, 6))

df_roll = df_roll.droplevel('HomeTeam')
df_roll.index = range(df_roll.shape[0])
home_dataframe = pd.get_dummies(df_roll['HomeTeam'])
df_roll = df_roll.join(home_dataframe)
home_col_list = list(home_dataframe.columns.values)
home_list = {}
encoded_h_list = []
encoded_a_list = []
for i in range(0, len(home_col_list)):
    home_list[home_col_list[i]] = f'Home_{home_col_list[i]}'
    encoded_h_list.append(f'Home_{home_col_list[i]}')
df_roll.rename(columns=home_list, inplace=True)

away_dataframe = pd.get_dummies(df_roll['AwayTeam'])
df_roll = df_roll.join(away_dataframe)
away_col_list = list(away_dataframe.columns.values)
away_list = {}
for i in range(0, len(away_col_list)):
    away_list[away_col_list[i]] = f'Away_{away_col_list[i]}'
    encoded_a_list.append(f'Away_{away_col_list[i]}')
print(df_roll.columns)
df_roll.rename(columns=away_list, inplace=True)
df_roll.drop('HomeTeam', axis=1, inplace=True)
df_roll.drop('AwayTeam', axis=1, inplace=True)
print(df_roll.columns)
# breakpoint()
train_data = df_roll[df_roll['Date'] < '2021-05-24']
test_data = df_roll[df_roll['Date'] > '2021-05-23']

train_data.index = range(train_data.shape[0])
test_data.index = range(test_data.shape[0])

features = new_cols + encoded_a_list + encoded_h_list + ["B365H", 'B365D', 'B365A']
target_data = ['FTR']
print_df = train_data.head(10)

print(train_data[features].to_string())
print(train_data['FTR'].to_string())
train_transformer = Normalizer(norm='max').fit(train_data[features])
train_data[features] = train_transformer.transform(train_data[features])

test_transformer = Normalizer(norm='max').fit(test_data[features])
test_data[features] = test_transformer.transform(test_data[features])

knn = RandomForestClassifier(n_estimators=10000, min_samples_split=10, random_state=0)
knn = DummyClassifier(strategy='most_frequent')
knn.fit(train_data[features], train_data['FTR'])
preds = knn.predict(test_data[features])
combined = pd.DataFrame(dict(actual=test_data["FTR"], predicted=preds), index=test_data.index)
# rf.score(d1_roll["FTR"], preds)
error = accuracy_score(test_data["FTR"], preds)
print(combined.to_string())
print(error)
cols_results = ['family', 'model', 'classification_rate', 'runtime']
results = pd.DataFrame(columns=cols_results)

kVals = range(1, 20)
X_train, X_test, y_train, y_test = train_test_split(train_data[features], train_data['FTR'], test_size=0.2,
                                                    random_state=1)
knn_names = ['KNN-' + str(k) for k in kVals]
for k in kVals:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    time_start = time.time()
    y_pred = knn.predict(X_test)
    time_run = time.time() - time_start

    results = results.append(
        pd.DataFrame([['KNN', knn_names[k - 1], accuracy_score(y_test, y_pred), time_run]], columns=cols_results),
        ignore_index=True)
var_1 = results[results.family == 'KNN']
print(var_1)

fig, ax = plt.subplots()
ax.plot(kVals, results[results.family == 'KNN'].classification_rate, color='blue', marker='o')
ax.set_xlabel('k-value for KNN models')
ax.set_ylabel('classification rate (blue)')
ax2 = ax.twinx()
ax2.plot(kVals, results[results.family == 'KNN'].runtime, color='red', marker='o')
ax2.set_ylabel('runtime (seconds; red)')
plt.show()

rVals = range(1, 4)
rf_names = ['RF-' + str(int(math.pow(10, r))) for r in rVals]

for r in rVals:
    clf = RandomForestClassifier(n_estimators=int(math.pow(10, r)), random_state=0)
    time_start = time.time()
    clf.fit(X_train, y_train)
    time_run = time.time() - time_start
    y_pred = clf.predict(X_test)

    results = results.append(
        pd.DataFrame([['RF', rf_names[r - 1], accuracy_score(y_test, y_pred), time_run]], columns=cols_results),
        ignore_index=True)

var_2 = results[results.family == 'RF']
print(var_2)

from sklearn.ensemble import BaggingClassifier

kernels = ['linear', 'rbf', 'sigmoid']
kernel_names = ['SVM-' + kernel for kernel in kernels]
import numpy as np

for i in range(len(kernels)):
    mean_error = []
    std_error = []
    Ci_range = [0.01, 0.1, 1, 5, 10, 25, 50, 100]
    for Ci in Ci_range:
        clf = SVC(kernel=kernels[i], random_state=0, C=Ci)

        time_start = time.time()

        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(clf, train_data[features], train_data['FTR'],
                                 cv=5)  # test train division will be inside the function. So using full data
        time_run = time.time() - time_start

        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if (Ci == 0.01):
            from sklearn.metrics import confusion_matrix

            conf_matrix_lg = confusion_matrix(y_test, y_pred)
            plot_2 = get_plot_fn()
            ax1 = plot_2.subplot()
            sns.heatmap(conf_matrix_lg, annot=True, fmt='g',
                        ax=ax1);  # annot=True to annotate cells, ftm='g' to disable scientific notation
            # labels, title and ticks
            ax1.set_xlabel('Predicted labels');
            ax1.set_ylabel('True labels');
            ax1.set_title('Confusion Matrix' + 'C = 0.01' + 'Kernel: ' + kernels[i]);
            ax1.xaxis.set_ticklabels(['Draw', 'Away Win', 'Home Win']);
            ax1.yaxis.set_ticklabels(['Draw', 'Away Win', 'Home Win']);
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
            ruc_plot.plot([0, 1], [0, 1], 'b--')
            ruc_plot.title('ROC Curve For ' + kernels[i])
            ruc_plot.xlabel('False positive rate')
            ruc_plot.ylabel('True positive rate')
            ruc_plot.legend([f'Class 0 (auc: {a})', f'Class 1 (auc: {w})', f'Class -1 (auc: {a})', 'Random Classifier'],
                            bbox_to_anchor=(1.04, 1),
                            loc="upper left", fontsize='6')
            ruc_plot.show()

    import matplotlib.pyplot as plt

    plt.rc('font', size=18);
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.title('C vs Mean Accuracy for ' + kernels[i] + 'kernel')
    plt.errorbar(Ci_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel('Ci');
    plt.ylabel('Mean Accuracy')
    plt.show()

# lasso
from sklearn.linear_model import RidgeClassifier
import numpy as np

mean_error = [];
std_error = []
c_range = [0.01, 0.1, 1, 10, 100, 500, 1000, 5000]

X = train_data[features].to_numpy()
y = train_data['FTR'].to_numpy()

for c in c_range:
    clf = RidgeClassifier(alpha=1 / (2 * c))
    temp = []
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        time_start = time.time()
        clf.fit(X[train], y[train])
        ypred = clf.predict(X[test])
        from sklearn.metrics import mean_squared_error

        temp.append(mean_squared_error(y[test], ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

import matplotlib.pyplot as plt

plt.title('C vs error in RIDGE')
plt.errorbar(c_range, mean_error, yerr=std_error)
plt.xlabel('C');
plt.ylabel('Mean square error')
plt.xlim((0, 10))
plt.show()