import json
import os
import pickle
import requests

from bs4 import BeautifulSoup
import pandas as pd


import understatapi as us_obj
#generate_set_of_teams()
session = requests.Session()
leagues = ["EPL"]
for league in us_obj.api.LeagueEndpoint(leagues, session=session):
         #print(league.get_team_data(season='2015'))
         d_dict = league.get_match_data(season='2015')
         d_json = json.dumps(d_dict)
         print(d_json)
         df = pd.read_json(d_json)
         print(df.to_string())
         #pd.read_json(league.get_match_data(season='2015'))

# a = us_obj.api.LeagueEndpoint(leagues, session=session)
# d_dict = a.get_match_data(season='2015')
# d_json = json.dumps(d_dict)
# print(d_json)



# pkl = pickle
# with open("scraping/teams_set/EPL_teams.pkl", "rb") as f:
#     object = pkl.load(f)
#
# df = pd.DataFrame(object)
# df.to_csv(r'file.csv')
