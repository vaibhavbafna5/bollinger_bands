import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import json
import pymongo
import os
import smtplib, ssl

from pathlib import Path
from email.message import EmailMessage
from dotenv import load_dotenv
from collections import deque

load_dotenv()

EMAIL_PW = os.environ.get('EMAIL_PASSWORD')
RECEIVER_EMAIL = os.environ.get('RECEIVER_EMAIL')
SENDER_EMAIL = os.environ.get('SENDER_EMAIL')
MONGO_URL = os.environ.get('MONGO_URL')

class NpEncoder(json.JSONEncoder):
    # used to help dump data safely
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class SuperTrendRunner():
    """
    Architecture:
    - initialization function to generate the initial super trend & upload to Mongo
    - runner function that runs daily to generate message
    """

    def __init__(self, ticker, json_path='', debug_mode=False, multiplier=2.5, rolling_period=14, initial_amount=10000,):
        self.ticker = ticker
        self.json_path = json_path
        self.multiplier = multiplier
        self.rolling_period = rolling_period
        self.initial_amount = initial_amount
        self.debug_mode = debug_mode

    def run(self):
        # TODO: stop being lazy and clean up the runner
        super_trend_df = self.generate_super_trend_for_ticker()

        start_date = super_trend_df.iloc[0].name
        end_date = super_trend_df.iloc[-1].name

        print(f"generating super trend for {self.ticker} starting {start_date} and ending {end_date}")
        print(super_trend_df.tail(15))

        self.chart_super_trend(super_trend_df)

        portfolio_over_time, percent_differences = self.simulate_portfolio_on_strategy(super_trend_df)
        super_trend_df['portfolio_values'] = portfolio_over_time
        super_trend_df['percentage_change'] = percent_differences

        benchmark_comparison_df, benchmark_df = self.benchmark_strategy(super_trend_df)
        benchmark_comparison_df.head(5)

        self.visualize_benchmark_against_strategy(super_trend_df, benchmark_df)

        if not self.debug_mode:
            trade_decision = self.get_trade_decision(super_trend_df)
            self.send_trade_decision(trade_decision)
            print(trade_decision)

            self.write_last_row_to_disk(super_trend_df)
            super_trend_df.to_pickle(f'./{self.ticker}_super_trend_df')

        return super_trend_df

    def generate_super_trend_for_ticker(self, period="6mo"):
        ticker_data = yf.Ticker(self.ticker).history(period=period)
        
        # calculate high low 
        ticker_data = self.generate_average_true_range(ticker_data)
        ticker_data = self.generate_basic_bands(ticker_data)
        ticker_data = self.generate_final_bands(ticker_data)
        
        return ticker_data

    def generate_average_true_range(self, ticker_data):
        # calculate high - low difference
        ticker_data['high-low'] = ticker_data['High'] - ticker_data['Low']
        
        # calculate high - previous close 
        ticker_data['Previous Close'] = ticker_data['Close'].shift(1)
        ticker_data['high-previous_close'] = ticker_data['High'] - ticker_data['Previous Close']
        
        # calculate low - previous close
        ticker_data['low-previous_close'] = ticker_data['Low'] - ticker_data['Previous Close']
        
        # calculate true range
        ticker_data['true_range'] = ticker_data[['high-low', 'high-previous_close', 'low-previous_close']].max(axis=1)
        
        # calculate average true range
        ticker_data['average_true_range'] = ticker_data['true_range'].rolling(window=self.rolling_period).mean()
        
        return ticker_data

    def generate_basic_bands(self, ticker_data):
        ticker_data['high_low_avg'] = (ticker_data['High'] + ticker_data['Low']) / 2
        ticker_data['lower_band'] = ticker_data['high_low_avg'] - (self.multiplier * ticker_data['average_true_range'])
        ticker_data['higher_band'] = ticker_data['high_low_avg'] + (self.multiplier * ticker_data['average_true_range'])
        
        return ticker_data

    def generate_final_bands(self, ticker_data):
        # initialization values 
        start_index = self.rolling_period
        
        final_upper_band = [ticker_data.iloc[i]['higher_band'] for i in range(0, start_index)]
        final_lower_band = [ticker_data.iloc[i]['lower_band'] for i in range(0, start_index)]
        super_trend = [ticker_data.iloc[i]['higher_band'] for i in range(0, start_index)]
        
        buy_or_sell = [None for i in range(0, start_index)]
        
        for i in range(start_index, len(ticker_data.index)):
            current = i
            previous = i - 1

            current_basic_upper_band = ticker_data.iloc[current]['higher_band']
            current_basic_lower_band = ticker_data.iloc[current]['lower_band']
            current_close = ticker_data.iloc[current]['Close']

            previous_final_upper_band = final_upper_band[-1]
            previous_final_lower_band = final_lower_band[-1]
            previous_super_trend = super_trend[-1]
            previous_close = ticker_data.iloc[previous]['Close']

            if current_basic_upper_band < previous_final_upper_band or previous_close > previous_final_upper_band:
                final_upper_band.append(current_basic_upper_band)
            else:
                final_upper_band.append(previous_final_upper_band)


            if current_basic_lower_band > previous_final_lower_band or previous_close < previous_final_lower_band:
                final_lower_band.append(current_basic_lower_band)
            else:
                final_lower_band.append(previous_final_lower_band)


            current_final_upper_band = final_upper_band[-1]
            current_final_lower_band = final_lower_band[-1]

            if previous_super_trend == previous_final_upper_band and current_close < current_final_upper_band:
                bs = False
                buy_or_sell.append(bs)
                super_trend.append(current_final_upper_band)

            elif previous_super_trend == previous_final_upper_band and current_close > current_final_upper_band:
                bs = True
                buy_or_sell.append(bs)
                super_trend.append(current_final_lower_band)

            elif previous_super_trend == previous_final_lower_band and current_close > current_final_lower_band:
                bs = True
                buy_or_sell.append(bs)
                super_trend.append(current_final_lower_band)

            elif previous_super_trend == previous_final_lower_band and current_close < current_final_lower_band:
                bs = False
                buy_or_sell.append(bs)
                super_trend.append(current_final_upper_band)

        ticker_data['super_trend'] = super_trend
        ticker_data['final_higher_band'] = final_upper_band
        ticker_data['final_lower_band'] = final_lower_band
        ticker_data['buy_or_sell'] = buy_or_sell
        
        ticker_data = ticker_data.iloc[1:]
        return ticker_data
        
    def chart_super_trend(self, super_trend_df):
        # get buy/sell lines 
        buys_line = []
        sells_line = []

        for i in range(0, len(super_trend_df.index)):
            value = super_trend_df.iloc[i]['super_trend']
            if super_trend_df.iloc[i]['buy_or_sell']:
                buys_line.append(value)
                sells_line.append(None)
            else:
                sells_line.append(value)
                buys_line.append(None)
            
        # extract data for candles
        candlestick = go.Candlestick(
            name='candles',
            x=super_trend_df.index,
            open=super_trend_df['Open'],
            high=super_trend_df['High'],
            low=super_trend_df['Low'],
            close=super_trend_df['Close']
        )

        # extract data for lower & higher bands
        lower_band = go.Line(
            name='lower_band',
            x=super_trend_df.index,
            y=super_trend_df['lower_band'],
            line=dict(color="purple"),
        )

        higher_band = go.Line(
            name='higher_band',
            x=super_trend_df.index,
            y=super_trend_df['higher_band'],
            line=dict(color="purple"),
        )

        # extract data for final bands
        final_upper_band_line = go.Line(
            name='final_upper_band',
            x=super_trend_df.index,
            y=super_trend_df['final_higher_band'],
            line=dict(color='green'),
        )

        final_lower_band_line = go.Line(
            name='final_lower_band',
            x=super_trend_df.index,
            y=super_trend_df['final_lower_band'],
            line=dict(color='red'),
        )

        # extract data for super trend overall line
        super_trend_line = go.Line(
            name='super_trend_line',
            x=super_trend_df.index,
            y=super_trend_df['super_trend'],
            line=dict(color='black')
        )

        buy_line = go.Scatter(
            name='buy_line',
            x=super_trend_df.index,
            y=buys_line,
            line=dict(color='purple')
        )

        sell_line = go.Scatter(
            name='sell_line',
            x=super_trend_df.index,
            y=sells_line,
            line=dict(color='pink')
        )

        data = [candlestick, final_upper_band_line, final_lower_band_line, super_trend_line, buy_line, sell_line]
        fig = go.Figure(
            data=data,
        )
        
        fig.update_layout(
            autosize=False,
            width=1200,
            height=800,)

        fig.show()

    def simulate_portfolio_on_strategy(self, super_trend_df):
        portfolio_vals = [self.initial_amount]
        percent_differences = [0.0]
        buying = False

        initial_amt = self.initial_amount
        
        for i in range(1, len(super_trend_df.index)):
            current = i
            previous = i - 1
            current_trade = super_trend_df.iloc[current]['buy_or_sell']
            previous_trade = super_trend_df.iloc[previous]['buy_or_sell']
            raw_percent_difference = 0
            
            # buy order
            if previous_trade == False and current_trade == True:
                buying = True
                
            # sell order
            elif previous_trade == True and current_trade == False:
                previous_price = super_trend_df.iloc[previous]['Close']
                current_price = super_trend_df.iloc[current]['Close']
                raw_percent_difference = ((current_price - previous_price) / previous_price)
                buying = False
                
            if buying:
                previous_price = super_trend_df.iloc[previous]['Close']
                current_price = super_trend_df.iloc[current]['Close']
                raw_percent_difference = ((current_price - previous_price) / previous_price)
                
            percent_difference = 1 + raw_percent_difference
            initial_amt = initial_amt * percent_difference

            percent_differences.append(raw_percent_difference * 100)
            portfolio_vals.append(initial_amt)
                
        return portfolio_vals, percent_differences

    def simulate_portfolio_strategy_with_slippage(self, super_trend_df, buy_slippage=1, sell_slippage=1):
        buying_periods = deque()
        current_day = 1
        previous_day = 0

        portfolio_vals = [self.initial_amount]
        initial_amt = self.initial_amount

        while current_day != len(super_trend_df.index):
            previous_day_row = super_trend_df.iloc[previous_day]
            current_day_row = super_trend_df.iloc[current_day]

            previous_price = previous_day_row['Close']
            current_price = current_day_row['Close']

            # buying
            if current_day_row['buy_or_sell'] == True and previous_day_row['buy_or_sell'] == False:
                buying_periods.append([current_day, -1])

            # selling
            elif current_day_row['buy_or_sell'] == False and previous_day_row['buy_or_sell'] == True:
                buying_periods[-1][1] = current_day

            previous_day += 1
            current_day += 1

        # adjusting the day bought/sold to account for slippage (e.g being late on buys/sells)
        for bp in buying_periods:
            bp[0] += buy_slippage
            bp[1] += sell_slippage

        # calculating percentage/price differentials by day 
        i = 1
        bp = buying_periods.popleft() if buying_periods else None
        percentage_differences = [0.0]

        while i != len(super_trend_df.index):
            raw_percentage = 0
            if bp:
                if i < bp[0]:
                    raw_percentage = 0
                elif bp[0] <= i <= bp[1]:
                    previous_price = super_trend_df.iloc[i - 1]['Close']
                    current_price = super_trend_df.iloc[i]['Close']
                    raw_percentage = ((current_price - previous_price) / previous_price)
                elif i > bp[0]:
                    bp = buying_periods.popleft() if buying_periods else None

            percentage_difference = 1 + raw_percentage
            initial_amt = initial_amt * percentage_difference

            percentage_differences.append(percentage_difference)
            portfolio_vals.append(initial_amt)

            i += 1

        return portfolio_vals, percentage_differences
        

    def benchmark_strategy(self, super_trend_df, benchmark_ticker='VTI'):
        start_date = super_trend_df.iloc[0].name
        end_date = super_trend_df.iloc[-1].name
        print(f"start date: {start_date}")
        print(f"end date: {end_date}", "\n")

        # get benchmark data & calculate portfolio change by simply holding
        benchmark_df = yf.Ticker(benchmark_ticker).history(period="max")
        benchmark_df = benchmark_df[start_date:end_date]

        benchmark_percent_differences = [0.0]
        b_portfolio_values = [self.initial_amount]

        for i in range(1, len(benchmark_df.index)):
            current = i
            previous = i - 1

            current_portfolio_value = benchmark_df.iloc[current]['Close']
            previous_portfolio_value = benchmark_df.iloc[previous]['Close']

            raw_percent_difference = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
            percent_difference = 1 + raw_percent_difference

            b_previous_portfolio_value = b_portfolio_values[-1]
            b_portfolio_values.append(percent_difference * b_previous_portfolio_value)
            benchmark_percent_differences.append(raw_percent_difference * 100)
            
        benchmark_df['portfolio_values'] = b_portfolio_values
        benchmark_df['percentage_change'] = benchmark_percent_differences

        # get strategy principal & total values
        total_return = super_trend_df.loc[end_date]['portfolio_values']
        b_total_return = benchmark_df.loc[end_date]['portfolio_values']
        
        # get benchmark principal & total values
        principal = benchmark_df.loc[start_date]['portfolio_values']
        b_principal = benchmark_df.loc[start_date]['portfolio_values']

        # cumulative return
        cumulative_return = ((total_return - principal) / principal)
        cumulative_return = cumulative_return * 100

        # get benchmark cumulative return
        b_cumulative_return = ((b_total_return - b_principal) / b_principal)
        b_cumulative_return = b_cumulative_return * 100

        # annual return
        num_years_invested = int(((end_date - start_date).days) / 365)
        annual_return = ((total_return/principal) ** (1/num_years_invested)) - 1
        annual_return = annual_return * 100

        # benchmark annual return
        num_years_invested = int(((end_date - start_date).days) / 365)
        b_annual_return = ((b_total_return/b_principal) ** (1/num_years_invested)) - 1
        b_annual_return = b_annual_return * 100
        
        # annual volatility
        percent_differences = list(super_trend_df['percentage_change'].values)
        mean = sum(percent_differences) / len(percent_differences)
        variance = sum([((x - mean) ** 2) for x in percent_differences]) / len(percent_differences)
        volatility = variance ** 0.5

        # benchmark annual volatility    
        b_mean = sum(benchmark_percent_differences) / len(benchmark_percent_differences)
        b_variance = sum([((x - b_mean) ** 2) for x in benchmark_percent_differences]) / len(benchmark_percent_differences)
        b_volatility = b_variance ** 0.5
        
        # sharpe ratio
        treasury_rate = 3.5
        sharpe_ratio = (annual_return - treasury_rate) / variance

        # benchmark sharpe ratio
        b_sharpe_ratio = (b_annual_return - treasury_rate) / b_variance
        
        # calmar ratio
        portfolio_values = super_trend_df['portfolio_values'].values
        max_portfolio_val = max(portfolio_values)
        min_portfolio_val = min(portfolio_values)
        drawdown = (max_portfolio_val - min_portfolio_val) / (max_portfolio_val)
        calmar_ratio = (annual_return - treasury_rate) / drawdown

        # benchmark calmar ratio
        b_portfolio_values = benchmark_df['portfolio_values'].values
        b_max_portfolio_val = max(b_portfolio_values)
        b_min_portfolio_val = min(b_portfolio_values)
        b_drawdown = (b_max_portfolio_val - b_min_portfolio_val) / b_max_portfolio_val
        b_calmar_ratio = (b_annual_return - treasury_rate) / b_drawdown
        
        benchmark_comparison_dict = {
            'principal' : [self.initial_amount, self.initial_amount],
            'total_return' : [total_return, b_total_return],
            'cumulative_return' : [cumulative_return, b_cumulative_return],
            'annual_return': [annual_return, b_annual_return],
            'volatility': [volatility, b_volatility],
            'sharpe_ratio': [sharpe_ratio, b_sharpe_ratio],
            'calmar_ratio': [calmar_ratio, b_calmar_ratio]
        }

        # creating a Dataframe object from dictionary 
        # with custom indexing
        benchmark_comparison_df = pd.DataFrame(benchmark_comparison_dict, index = ['strategy', 'benchmark',])
        return benchmark_comparison_df, benchmark_df

    def visualize_benchmark_against_strategy(self, super_trend_df, benchmark_df):
        benchmark_line = go.Line(
            name='benchmark_VTI',
            x=benchmark_df.index,
            y=benchmark_df['portfolio_values'],
            line=dict(color='red'),
        )

        strategy_line = go.Line(
            name=f'super_trend_strategy_{self.ticker}',
            x=super_trend_df.index,
            y=super_trend_df['portfolio_values'],
            line=dict(color='green'),
        )

        fig = go.Figure(
            data=[benchmark_line, strategy_line]
        )

        fig.update_layout(
            autosize=False,
            width=1200,
            height=800,)

        fig.show()

    def get_trade_decision(self, super_trend_df):
        last_trading_day = super_trend_df.iloc[-1]
        second_to_last_trading_day = super_trend_df.iloc[-2]
        
        # buy order
        if second_to_last_trading_day['buy_or_sell'] == False and last_trading_day['buy_or_sell'] == True:
            msg = f"Buy {self.ticker}"
        # sell order
        elif second_to_last_trading_day['buy_or_sell'] == True and last_trading_day['buy_or_sell'] == False:
            msg = f"Sell {self.ticker}"
        # maintain status quo
        else:
            msg = f"No-Op {self.ticker}"

        last_trading_day_date = str(last_trading_day.name.date())
        return msg + " " + last_trading_day_date

    def send_trade_decision_via_email(self, trade_decision):
        port = 587
        smtp_server = "smtp.gmail.com"

        msg = EmailMessage()
        body = ""
        msg.set_content(body)

        msg['Subject'] = trade_decision
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL

        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(SENDER_EMAIL, EMAIL_PW)
            server.send_message(msg)

    def write_last_row_to_disk(self, super_trend_df):
        last_row = super_trend_df.iloc[-1]

        last_row_as_dict = last_row.to_dict()
        last_row_time = str(last_row.name.date())
        
        base = Path(f'{self.json_path}/{self.ticker}')
        jsonpath = base / (last_row_time + ".json")
        base.mkdir(exist_ok=True)
        jsonpath.write_text(json.dumps(last_row_as_dict, cls=NpEncoder))
        print(last_row_as_dict)

    def initialize_dataframe(self, period="6mo"):
        super_trend_df = self.generate_super_trend_for_ticker(period)

        start_date = super_trend_df.iloc[0].name
        end_date = super_trend_df.iloc[-1].name

        print(f"generating super trend for {self.ticker} starting {start_date} and ending {end_date}")

        portfolio_over_time, percent_differences = self.simulate_portfolio_on_strategy(super_trend_df)
        super_trend_df['portfolio_values'] = portfolio_over_time
        super_trend_df['percentage_change'] = percent_differences

        return super_trend_df

    def write_initial_dataframe_to_mongo(self, super_trend_df):
        client = pymongo.MongoClient(MONGO_URL)
        db = client['super-trend']

        if self.ticker not in db.list_collection_names():
            print(f"{self.ticker} is not a collection in DB")
            return

        collection = db[self.ticker]

        super_trend_df.reset_index(inplace=True)
        values = list(super_trend_df.T.to_dict().values())
        collection.insert_many(values)

    def execute_daily_trade_decision(self):
        super_trend_df = self.generate_daily_super_trend()
        trade_decision = self.get_trade_decision(super_trend_df)
        print("trade decision: ", trade_decision)

        if not self.debug_mode:
            self.write_trade_decision_to_mongo(super_trend_df)
            self.send_trade_decision_via_email(trade_decision)

    def generate_daily_super_trend(self, period="6mo"):
        super_trend_df = self.generate_super_trend_for_ticker(period=period)
        portfolio_over_time, percent_differences = self.simulate_portfolio_on_strategy(super_trend_df)
        super_trend_df['portfolio_values'] = portfolio_over_time
        super_trend_df['percentage_change'] = percent_differences

        return super_trend_df

    def write_trade_decision_to_mongo(self, super_trend_df):
        super_trend_df.reset_index(inplace=True)
        last_row = super_trend_df.iloc[-1].to_dict()

        client = pymongo.MongoClient(MONGO_URL)
        db = client['super-trend']

        if self.ticker not in db.list_collection_names():
            print(f"{self.ticker} is not a collection in DB")
            return

        collection = db[self.ticker]
        collection.insert_one(last_row)

    def read_trade_data_from_mongo_to_dataframe(self):
        client = pymongo.MongoClient(MONGO_URL)
        db = client['super-trend']

        if self.ticker not in db.list_collection_names():
            print(f"{self.ticker} is not a collection in DB")
            return

        collection = db[self.ticker]
        cursor = collection.find()
        list_cur = list(cursor)

        df = pd.DataFrame(list_cur)
        return df
        