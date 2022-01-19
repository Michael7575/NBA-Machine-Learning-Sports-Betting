import requests
import pandas as pd
from datetime import datetime
from pycaret.classification import *

games_header = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/57.0.2987.133 Safari/537.36',
    'Dnt': '1',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'en',
    'origin': 'http://stats.nba.com',
    'Referer': 'https://github.com'
}

data_headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.4 Safari/605.1.15',
    'Accept-Language': 'en-us',
    'Referer': 'https://stats.nba.com/teams/traditional/?sort=W_PCT&dir=-1&Season=2019-20&SeasonType=Regular%20Season',
    'Connection': 'keep-alive',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true'
}


def get_json_data(url):
    raw_data = requests.get(url, headers=data_headers)
    json = raw_data.json()
    return json.get('resultSets')


def get_todays_games_json(url):
    raw_data = requests.get(url, headers=games_header)
    json = raw_data.json()
    return json.get('gs').get('g')

def to_data_frame(data):
    data_list = data[0]
    return pd.DataFrame(data=data_list.get('rowSet'), columns=data_list.get('headers'))


def create_todays_games(input_list):
    games = []
    for game in input_list:
        home = game.get('h')
        away = game.get('v')
        home_team = home.get('tc') + ' ' + home.get('tn')
        away_team = away.get('tc') + ' ' + away.get('tn')
        
        year = game['gcode'][:4]
        month = game['gcode'][4:6]
        day = game['gcode'][6:8]
        dt = f'{year}-{month}-{day}'
        
        try:
            st = game['stt']
            st_timezone = st[-2:]
            st_time = st[:-3]    
            dt_string = f'{dt} {st_time}'  
            timestamp = datetime.strptime(dt_string, "%Y-%m-%d %I:%M %p")
        except:
            st = 'Game has started'
            timestamp = datetime.strptime(dt, "%Y-%m-%d")
            st_timezone = 'ET'
        
        games.append({'home_team':home_team,
                      'away_team':away_team,
                      'date': dt,
                      'start_time': st,
                      'timestamp': timestamp,
                      'timezone': st_timezone})
    return games

def create_seasons_dict(start_year,end_year):
    
    num_years=end_year-start_year
    
    seasons_dicts=[]
    
    for i in range(num_years):
        
        season_start_year=start_year+i
        season_end_year=season_start_year+1
        
        start_year_str=str(season_start_year)
        
        years=[season_start_year,season_end_year]
        
        start_year_str=str(season_start_year)
        end_year_str=str(season_end_year)[-2:]
        season=f'{start_year_str}-{end_year_str}'
        
        dic={"season":season,
             "years":years}
        
        seasons_dicts.append(dic)
    
    return seasons_dicts

def expected_value(Pwin, odds):
    """
    In betting, the expected value (EV) is the measure of what a bettor 
    can expect to win or lose per bet placed on the same odds time and time again. 
    Positive expected value (+EV) implies profit over time, 
    while a negative value (-EV) implies a loss over time.
    
    """
    Ploss = 1 - Pwin
    Mwin = payout(odds)
    return round((Pwin * Mwin) - (Ploss * 100), 2)


def payout(odds):
    if odds > 0:
        return odds
    else:
        return (100 / (-1 * odds)) * 100
    
def get_expected_values(pred_df):
    
    pred_dict = pred_df.to_dict(orient='records')
    
    for game in pred_dict:

        Pwin_home = round(game['Score_W'],4)
        Pwin_away = round(game['Score_L'],4)

        odds_home = int(game['ml_home'])
        odds_away = int(game['ml_away'])

        home_team_ev = expected_value(Pwin_home, odds_home)
        away_team_ev = expected_value(Pwin_away, odds_away)

        game['home_team_ml_expected_value'] = home_team_ev
        game['away_team_ml_expected_value'] = away_team_ev
        
    return pred_dict

def predict(games):
    
    if isinstance(games, list):
        pred_df = pd.DataFrame(games)
        
    else:
        
        pred_df = games
        
    pred_df.columns = pred_df.columns.str.lower()

    #load models
    win_loss_model = load_model('win_loss_acc_72')
    ou_model = load_model('ou_cover_acc_56')
    
    #make predictions
    win_loss_prediction_df = predict_model(win_loss_model, data = pred_df, raw_score = True)
    ou_prediction_df = predict_model(ou_model, data = pred_df, raw_score = True)
    
    #Get ML expect values
    win_loss_results = get_expected_values(win_loss_prediction_df)
    ou_results = ou_prediction_df.to_dict(orient='records')
    
    return win_loss_results, ou_results

def clean_train_data(raw_df):
    
    raw_df.drop(['Win_Margin', 'Unnamed: 0'], axis=1, errors='ignore',inplace=True)
    
    raw_df.columns=raw_df.columns.str.lower()
    
    raw_df=raw_df[~raw_df['ml_home'].isin(["NL"])]
    
    raw_df['ml_home']=raw_df['ml_home'].astype(float)
    raw_df['ml_away']=raw_df['ml_away'].astype(float)
    
    return raw_df

