from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append("/content/drive/MyDrive/kubig_financial/finrl")

# ## install required packages
# !pip install gym
# !pip install stockstats
# !pip install gymnasium
# !pip install stable_baselines3
# !pip install alpaca_trade_api
# !pip install exchange_calendars
# !pip install pyfolio
# !pip install matplotlib
# !pip install swig
# !pip install wrds
# !pip install pyportfolioopt
# !pip install --upgrade ipykernel
# ## install finrl library
# !apt-get update -y -qq && apt-get install -y -qq cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig

def finrl_main(tickers, using, epochs, FORCAST_SIZE, tolerance, hmax, initial_amount):


  import sys
  sys.path.append("/content/drive/MyDrive/kubig_financial/finrl")

  import pandas as pd
  import numpy as np
  import matplotlib
  import matplotlib.pyplot as plt
  # matplotlib.use('Agg')
  import datetime
  from stable_baselines3 import A2C

  # %matplotlib inline
  from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
  from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
  from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
  from finrl.agents.stablebaselines3.models import DRLAgent
  from stable_baselines3.common.logger import configure
  from finrl.meta.data_processor import DataProcessor

  from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
  from pprint import pprint
  
  import itertools

  from finrl import config
  from finrl import config_tickers
  import os
  from finrl.main import check_and_make_directories
  from finrl.config import (
      DATA_SAVE_DIR,
      TRAINED_MODEL_DIR,
      TENSORBOARD_LOG_DIR,
      RESULTS_DIR,
      INDICATORS,
      TRAIN_START_DATE,
      TRAIN_END_DATE,
      TEST_START_DATE,
      TEST_END_DATE,
      TRADE_START_DATE,
      TRADE_END_DATE,
  )
  check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

  # %cd /content/drive/MyDrive/kubig_financial
  # !ls
  import dlinear_module

  import datetime
  TRADE_END_DATE = datetime.date.today().strftime("%Y-%m-%d")

  ##### parameter inside #####
  TRAIN_START_DATE = '2009-01-01'
  TRAIN_END_DATE = '2021-10-01'
  TRADE_START_DATE = '2021-10-01'

  # TRADE_END_DATE = '2023-07-01'
  tickers = tickers

  ## agent setting
  if_using_a2c = False
  if_using_ddpg = False
  if_using_ppo = False
  if_using_td3 = False
  if_using_sac = False

  if using == 'a2c':
    if_using_a2c = True
    using_stratgy = if_using_a2c
  elif using == 'ddpg':
    if_using_ddpg = True
    using_stratgy = if_using_ddpg
  elif using == 'ppo':
    if_using_ppo = True
    using_stratgy = if_using_ppo
  elif using == 'td3':
    if_using_td3 = True
    using_stratgy = if_using_td3
  elif using == 'sac':
    if_using_td3 = True
    using_stratgy = if_using_sac

  ############################
  print('start_date : ',TRAIN_START_DATE,'end_date : ',TRADE_END_DATE,'tickers : ',tickers)
  ## data loading
  df = YahooDownloader(start_date = TRAIN_START_DATE, end_date = TRADE_END_DATE, ticker_list = tickers).fetch_data()
  df_yahoo = df.copy()
  df.sort_values(['date','tic'],ignore_index=True)

  ## load dlinear prediction
  prediction_5,prediction_10,prediction_30 = dlinear_module.main(FORCAST_SIZE, epochs, tolerance, TRADE_END_DATE, df=df_yahoo, tickers=tickers)

  print(INDICATORS)
  ## feature Engineering
  fe = FeatureEngineer(use_technical_indicator=True,
                      tech_indicator_list = INDICATORS,
                      use_vix=True,
                      use_turbulence=True,
                      user_defined_feature = False)

  processed = fe.preprocess_data(df)
  processed.sort_values(['date','tic'],ignore_index=True)

  list_ticker = processed["tic"].unique().tolist() #ticker 리스트 불러오기
  list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str)) #전체 데이터 날짜 날짜 리스트
  combination = list(itertools.product(list_date,list_ticker)) #date, ticker 의 combination

  processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left") #date, tic combination 기준으로 우측에 해당되는 정보들 정리
  processed_full = processed_full[processed_full['date'].isin(processed['date'])]
  processed_full = processed_full.sort_values(['date','tic'])
  processed_full = processed_full.fillna(0)

  ## combine prediction
  prediction_5_list = []
  for i in range(len(prediction_5)):
      for j in range(len(tickers)):
          prediction_5_list.append(prediction_5.iloc[i,j])

  for i in range((len(processed_full)-len(prediction_5_list))//len(tickers)):
      for j in range(len(tickers)):
          prediction_5_list.append(prediction_5.iloc[-1,j])

  prediction_10_list = []
  for i in range(len(prediction_10)):
      for j in range(len(tickers)):
          prediction_10_list.append(prediction_10.iloc[i,j])

  for i in range((len(processed_full)-len(prediction_10_list))//len(tickers)):
      for j in range(len(tickers)):
          prediction_10_list.append(prediction_10.iloc[-1,j])

  prediction_30_list = []
  for i in range(len(prediction_30)):
      for j in range(len(tickers)):
          prediction_30_list.append(prediction_30.iloc[i,j])
  for i in range((len(processed_full)-len(prediction_30_list))//len(tickers)):
      for j in range(len(tickers)):
          prediction_30_list.append(prediction_30.iloc[-1,j])

  processed_full['prediction_5']=prediction_5_list
  processed_full['prediction_10']=prediction_10_list
  processed_full['prediction_30']=prediction_30_list

  INDICATORS += ['prediction_5','prediction_10','prediction_30']

  ## data split
  train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
  trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)

  stock_dimension = len(train.tic.unique())
  state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension #현재 상태를 나타내는 state를 정의. 1+ ticker 개수 x2 + 인디케이터 개수 x ticker개수
  print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

  buy_cost_list = sell_cost_list = [0.001] * stock_dimension
  num_stock_shares = [0] * stock_dimension

  env_kwargs = {
      "hmax": hmax,
      "initial_amount": initial_amount,
      "num_stock_shares": num_stock_shares,
      "buy_cost_pct": buy_cost_list,
      "sell_cost_pct": sell_cost_list,
      "state_space": state_space,
      "stock_dim": stock_dimension,
      "tech_indicator_list": INDICATORS,
      "action_space": stock_dimension,
      "reward_scaling": 1e-4
  }

  e_train_gym = StockTradingEnv(df = train, **env_kwargs)

  env_train, _ = e_train_gym.get_sb_env()


  ## agent setting
  agent = DRLAgent(env = env_train)
  model_using = agent.get_model(f"{using}")

  if using_stratgy:
    # set up logger
    tmp_path = RESULTS_DIR + f'{using}'
    new_logger_using = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_using.set_logger(new_logger_using)

  if tickers != ['XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']:
    ## training
    trained_using = agent.train_model(model=model_using,
                                tb_log_name=f'{using}',
                                total_timesteps=50000) if using_stratgy else None
  else: # ticker가 default값이면
    path = '/content/drive/MyDrive/kubig_financial/result/trained_a2c_model.pt'
    trained_using = A2C.load(path)

  data_risk_indicator = processed_full[(processed_full.date<TRAIN_END_DATE) & (processed_full.date>=TRAIN_START_DATE)]
  insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=['date'])

  e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)

  trained_model = trained_using
  df_account_value_using, df_actions_using = DRLAgent.DRL_prediction(
      model=trained_model,
      environment = e_trade_gym)

  df_result_using = df_account_value_using.set_index(df_account_value_using.columns[0])
  df_account_value_using.to_csv(f"/content/drive/MyDrive/kubig_financial/result/df_account_value_{using}.csv")
  #baseline stats
  print("==============Get Baseline Stats===========")
  df_dji_ = get_baseline(
          ticker="^DJI",
          start = TRADE_START_DATE,
          end = TRADE_END_DATE)
  stats = backtest_stats(df_dji_, value_col_name = 'close')
  df_dji = pd.DataFrame()
  df_dji['date'] = df_account_value_using['date']
  df_dji['account_value'] = df_dji_['close'] / df_dji_['close'][0] * env_kwargs["initial_amount"]
  # df_dji.to_csv("df_dji.csv")
  df_dji = df_dji.set_index(df_dji.columns[0])
  df_dji.to_csv("/content/drive/MyDrive/kubig_financial/result/df_dji+.csv")


  ########## result ##########
  result = pd.DataFrame(df_result_using)

  """result = pd.merge(df_result_a2c, df_result_ddpg, left_index=True, right_index=True)
  result = pd.merge(result, df_result_td3, left_index=True, right_index=True)
  result = pd.merge(result, df_result_ppo, left_index=True, right_index=True)
  result = pd.merge(result, df_result_sac, left_index=True, right_index=True)
  result = pd.merge(result, MVO_result, left_index=True, right_index=True)"""
  result = pd.merge(result, df_dji, left_index=True, right_index=True)


  result.columns = [f'{using}','dji']

  # print("result: ", result)
  result.to_csv(f"/content/drive/MyDrive/kubig_financial/result/result_{using}.csv")

  ## visualize
  # %matplotlib inline
  plt.rcParams["figure.figsize"] = (15,5)
  plt.figure();
  result.plot();

  ## 수익률
  print(f'{using} 로 얻은 투자 수익률 >>', round((df_result_using.iloc[-1,0]/df_result_using.iloc[0,0]-1)*100,2),'%')
  ## 보유현금
  df_daily_return = df_account_value_using
  ## 거래내역
  df_actions_using

  return round((df_result_using.iloc[-1,0]/df_result_using.iloc[0,0]-1)*100,2), df_daily_return, df_actions_using