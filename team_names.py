import pandas as pd

year_list = [2021, 2020, 2019, 2018, 2017, 2016, 2015]
len_year = len(year_list)

season_wise_teams = {}

def get_unique_team_names():
    team_set = set()
    fb_co_data_list = []
    fb_ref_list = []
    for i in range(0, len_year):
        print(i)
        print(year_list)
        df = pd.read_csv(f'fb_co_data_{year_list[i]}.csv')
        df = df.replace({'HomeTeam': team_name_dict})
        team_name = df['HomeTeam'].replace(r'\n',' ', regex=True).tolist()
        team_set.update(team_name)
        #print(team_name_set)
    for i in range(0,len_year):
        df = pd.read_csv(f'fb_ref_data_{year_list[i]}.csv')
        df = df.replace({'Home': team_name_dict})
        team_name = df['Home'].replace(r'\n',' ', regex=True).tolist()
        team_set.update(team_name)

    #team_set = set(team_list)
    return team_set

def get_team_name(csv_name, year):

    df = pd.read_csv(csv_name)
    team_list = df['HomeTeam'].tolist()
    team_set = set(team_list)
    season_wise_teams[year] =  list(team_set)




for i in range(0, len_year):
    csv_name = f'final_data/final_csv_{i}.csv'
    get_team_name(csv_name, year_list[i])
#print(season_wise_teams)



team_name_dict = {
    'Leeds': 'Leeds United',
    'Leeds United': 'Leeds United',
    'Norwich City': 'Norwich City',
    'Norwich': 'Norwich City',
    'Man City': 'Manchester City',
    'Man United': 'Manchester United',
    'Manchester Utd': 'Manchester United',
    'Manchester City': 'Manchester City',
    'Sheffield': 'Sheffield United',
    'Sheffield Utd': 'Sheffield United',
    'Newcastle Utd': 'Newcastle United',
    'Newcastle': 'Newcastle United',
    'Hull': 'Hull City',
    'Hull City': 'Hull City',
    'Stoke': 'Stoke City',
    'Stoke City': 'Stoke City',
    'Swansea': 'Swansea City',
    'Swansea City': 'Swansea City',
    'Leicester': 'Leicester City',
    'Leicester City': 'Leicester City',
    'Cardiff': 'Cardiff City',
    'Cardiff City': 'Cardiff City',
}

# def get_teams_to_remove(year):
#     print(year)
#     a = season_wise_teams.get(year)
#     b = season_wise_teams.get(year-1)
#
#     a, b = [i for i in a if i not in b], [j for j in b if j not in a]
#     print(a)
#     print(b)
#     return a, b
#
# for i in range(0, len_year):
#     get_teams_to_remove(year_list[i])
#print(get_unique_team_names())
#print(len(get_unique_team_names()))

# df = pd.read_csv(f'fb_co_data_0.csv')
# df1 = pd.read_csv(f'fb_ref_data_0.csv')
# #team_name = df['HomeTeam'].replace(r'\n',' ', regex=True)
# team_name_norm = df.replace({'HomeTeam': team_name_dict})
# team_ref_norm = df1.replace({'Home': team_name_dict})
# print(team_ref_norm['Home'])
# print(team_name_norm['HomeTeam'])


df = pd.read_csv(f'fb_co_data_2017.csv')
print(df['FTR'].value_counts())