
import pandas as pd
year_list = [2021, 2020, 2019, 2018, 2017, 2016, 2015]

fb_ref_links = [
    'https://fbref.com/en/comps/9/2021-2022/schedule/2021-2022-Premier-League-Scores-and-Fixtures',
    'https://fbref.com/en/comps/9/2020-2021/schedule/2020-2021-Premier-League-Scores-and-Fixtures',
    'https://fbref.com/en/comps/9/2019-2020/schedule/2019-2020-Premier-League-Scores-and-Fixtures',
    'https://fbref.com/en/comps/9/2018-2019/schedule/2018-2019-Premier-League-Scores-and-Fixtures',
    'https://fbref.com/en/comps/9/2017-2018/schedule/2017-2018-Premier-League-Scores-and-Fixtures',
    'https://fbref.com/en/comps/9/2016-2017/schedule/2016-2017-Premier-League-Scores-and-Fixtures',
    'https://fbref.com/en/comps/9/2015-2016/schedule/2015-2016-Premier-League-Scores-and-Fixtures',
]
# URL = "https://fbref.com/en/comps/9/2021-2022/schedule/2021-2022-Premier-League-Scores-and-Fixtures"
football_co_data = [
    'https://www.football-data.co.uk/mmz4281/2122/E0.csv',
    'https://www.football-data.co.uk/mmz4281/2021/E0.csv',
    'https://www.football-data.co.uk/mmz4281/1920/E0.csv',
    'https://www.football-data.co.uk/mmz4281/1819/E0.csv',
    'https://www.football-data.co.uk/mmz4281/1718/E0.csv',
    'https://www.football-data.co.uk/mmz4281/1617/E0.csv',
    'https://www.football-data.co.uk/mmz4281/1516/E0.csv',

]


def get_fb_data(url_list, csv_file_name, file_type):
    count = 0

    for url in url_list:

        count_name = count
        if file_type == 'html':
            print(url)
            df = pd.read_html(url)[0]
            #df.drop(columns='Unnamed: 0', axis=1, inplace=True)
            df.to_csv(f'{csv_file_name}_{year_list[count_name]}.csv')
        else:
            df = pd.read_csv(url)
            #df.drop(columns='Unnamed: 0', axis=1, inplace=True)
            df.to_csv(f'{csv_file_name}_{year_list[count_name]}.csv')
        count = count + 1




get_fb_data(fb_ref_links, 'fb_ref_data', 'html')
get_fb_data(football_co_data, 'fb_co_data', 'csv')