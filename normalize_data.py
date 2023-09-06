import pandas as pd

def create_normalised_csv():
    return ""


ars_df = pd.read_csv('final_data/final_csv_0.csv')
print(ars_df.columns)
ars_df = ars_df.loc[(ars_df['HomeTeam'] == 'Arsenal') | (ars_df['AwayTeam'] == 'Arsenal') ]
print(ars_df.to_string())

