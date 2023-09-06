import math
import time

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


def rolling_averages(group, cols, new_cols, roll_match):
    group = group.sort_values("Date")
    #zscore = lambda x: (x.values[-1] - x.mean()) / x.std(ddof=1)
    #rolling_stats = (group[cols] - group[cols].rolling(roll_match, closed='left')) /group[cols].rolling(roll_match).std()
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




cols = ["FTHG", "FTAG",'ref_xg_home', 'ref_xg_away', 'FTR']
new_cols = [f"{c}_avg" for c in cols]

df = pd.read_csv(f'final_df.csv')

df_roll = df.groupby('HomeTeam').apply(lambda x: rolling_averages(x, cols, new_cols,6))

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
#breakpoint()
train_data = df_roll[df_roll['Date'] < '2021-05-24']
test_data = df_roll[df_roll['Date'] > '2021-05-23']

train_data.index = range(train_data.shape[0])
test_data.index = range(test_data.shape[0])


features = new_cols + encoded_a_list + encoded_h_list + ["B365H",'B365D','B365A']
target_data = ['FTR']
print_df = train_data.head(10)


print(train_data[features].to_string())
print(train_data['FTR'].to_string())
train_transformer = Normalizer(norm='max').fit(train_data[features])
train_data[features] = train_transformer.transform(train_data[features])

test_transformer = Normalizer(norm='max').fit(test_data[features])
test_data[features] = test_transformer.transform(test_data[features])



knn = RandomForestClassifier(n_estimators=10000, min_samples_split=10, random_state=0)
#knn = DummyClassifier(strategy='most_frequent')
knn.fit(train_data[features], train_data['FTR'])
preds = knn.predict(test_data[features])
combined = pd.DataFrame(dict(actual=test_data["FTR"], predicted=preds), index=test_data.index)
#rf.score(d1_roll["FTR"], preds)
error = accuracy_score(test_data["FTR"], preds)
print(combined.to_string())
print(error)
cols_results=['family','model','classification_rate','runtime']
results = pd.DataFrame(columns=cols_results)
featuresz=[]
for i in range(0,len(features)):
    if(knn.feature_importances_[i] >0.05):
        featuresz.append(features[i])
#breakpoint()
kVals = range(1, 20)
X_train, X_test, y_train, y_test = train_test_split(train_data[features],train_data['FTR'], test_size=0.2, random_state=1)
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
ax.plot(kVals, results[results.family =='KNN'].classification_rate,color='blue',marker='o')
ax.set_xlabel('k-value for KNN models')
ax.set_ylabel('classification rate (blue)')
ax2= ax.twinx()
ax2.plot(kVals, results[results.family=='KNN'].runtime,color='red',marker='o')
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

kernels = ['linear', 'rbf', 'sigmoid']
kernel_names = ['SVM-' + kernel for kernel in kernels]

for i in range(len(kernels)):
    clf = SVC(kernel=kernels[i], random_state=0)
    time_start = time.time()
    #clf = BaggingClassifier(base_estimator=clf, n_estimators=10, max_samples=0.01, n_jobs=-1, random_state=0)
    clf.fit(X_train, y_train)
    time_run = time.time() - time_start
    y_pred = clf.predict(X_test)

    results = results.append(
        pd.DataFrame([['SVM', kernel_names[i], accuracy_score(y_test, y_pred), time_run]], columns=cols_results),
        ignore_index=True)

var_3 = results[results.family == 'SVM']
print(var_3)