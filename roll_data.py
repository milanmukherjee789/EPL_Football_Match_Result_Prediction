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

def rolling_averages(grouped_team, orig_col, new_col, roll_match):
    sorted_group_val = grouped_team.sort_values("Date")
    #zscore = lambda x: (x.values[-1] - x.mean()) / x.std(ddof=1)
    #rolling_stats = (group[cols] - group[cols].rolling(roll_match, closed='left')) /group[cols].rolling(roll_match).std()
    rolling_stats = sorted_group_val[orig_col].rolling(roll_match, closed='left').mean()
    sorted_group_val[new_col] = rolling_stats
    sorted_group_val = sorted_group_val.dropna(subset=new_col)
    return sorted_group_val


def get_team_form(group, cols, new_col, roll_match):
    group = group.sort_values('Date')
    form_match = group[cols].rolling(roll_match, closed='left').apply(lambda x: x.mode()[0])
    print(form_match)
    group[new_col] = form_match
    group = group.dropna(subset=new_col)
    return group



def rolled_data(roll):
    cols = ["FTHG", "FTAG", 'ref_xg_home', 'ref_xg_away', 'FTR']
    new_cols = [f"{c}_avg" for c in cols]

    df = pd.read_csv(f'final_df.csv')

    df_roll = df.groupby('HomeTeam').apply(lambda x: rolling_averages(x, cols, new_cols, roll))

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
    train_transformer = Normalizer(norm='max').fit(train_data[features])
    train_data[features] = train_transformer.transform(train_data[features])

    test_transformer = Normalizer(norm='max').fit(test_data[features])
    test_data[features] = test_transformer.transform(test_data[features])
    return train_data[features], test_data[features], train_data['FTR'], test_data['FTR']

