import logging
import os
import pandas as pd
import plotly.graph_objects as go
import sys
import yfinance as yf

from collections import deque
from datetime import datetime
from dotenv import load_dotenv
from IPython.display import display
from utils import mongo_instance, email_sender

load_dotenv()

RECEIVER_EMAIL = os.environ.get('RECEIVER_EMAIL')
SENDER_EMAIL = os.environ.get('SENDER_EMAIL')

# configure basic settings for the logging system
log_format = '[%(asctime)s] [%(levelname)s] [%(message)s] [%(filename)s:%(lineno)d]'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)  # set logging to output to standard output.
    ]
)

class BollingerBands:
    """"""

    def __init__(self, ticker, debug_mode=False, multiplier=2.5, rolling_period=14, initial_amount=10000,):
        """Initialize BollingerBands."""

        self.ticker = ticker  # asset to trade
        self.multiplier = multiplier  # multiplier for the average true range
        self.rolling_period = rolling_period  # number of days to take a rolling avg
        self.initial_amount = initial_amount  # initial portfolio amount
        self.debug_mode = debug_mode  # controls whether mongo is written to & email is sent


    def explore(self, period="6mo"):
        """
        - Generates the `bollinger_bands` dataframe
        - Charts the different bands
        - Simulates the portfolio earnings using this strategy
        - Benchmarks the strategy against holding a default ticker (normally VTI)
        - Renders key metrics for the strategy & the benchmark
        - Visualizes portfolio earnings for the strategy & the benchmark
        """

        # generate bollinger bands for ticker
        bollinger_bands_df = self.generate_bollinger_bands_for_ticker(period)
        start_date = bollinger_bands_df.iloc[0].name
        end_date = bollinger_bands_df.iloc[-1].name

        logging.info(f"Generating bollinger bands for {self.ticker} starting {start_date} and ending {end_date}")
        display(bollinger_bands_df.tail(15))

        # chart the trend
        self.chart_bollinger_band(bollinger_bands_df)

        # simulate the portfolio earnings using this strategy
        portfolio_over_time, percent_differences = self.simulate_portfolio_on_strategy(bollinger_bands_df)
        bollinger_bands_df['portfolio_values'] = portfolio_over_time
        bollinger_bands_df['percentage_change'] = percent_differences

        # benchmark the strategy against holding an alternative, e.g VTI 
        benchmark_comparison_df, benchmark_df = self.benchmark_strategy(bollinger_bands_df)

        # render key metrics for strategy & benchmark
        display(benchmark_comparison_df.head(5))

        # chart the strategy and benchmark portfolio holdings
        self.visualize_benchmark_against_strategy(bollinger_bands_df, benchmark_df)

        return bollinger_bands_df

        
    def generate_bollinger_bands_for_ticker(self, period):
        """Wraps generation of the `bands`."""

        ticker_data = yf.Ticker(self.ticker).history(period=period)
        ticker_data = self.generate_average_true_range(ticker_data)
        ticker_data = self.generate_basic_bands(ticker_data)
        ticker_data = self.generate_final_bands(ticker_data)
        
        return ticker_data


    def generate_average_true_range(self, ticker_data):
        """
        Generate average true range.
        See more here: https://en.wikipedia.org/wiki/Average_true_range
        """
        
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
        """Generate the basic bands based on the specified multiplier."""

        ticker_data['high_low_avg'] = (ticker_data['High'] + ticker_data['Low']) / 2
        ticker_data['lower_band'] = ticker_data['high_low_avg'] - (self.multiplier * ticker_data['average_true_range'])
        ticker_data['higher_band'] = ticker_data['high_low_avg'] + (self.multiplier * ticker_data['average_true_range'])
        
        return ticker_data


    def generate_final_bands(self, ticker_data):
        """Generate final bands."""

        # initialization values 
        start_index = self.rolling_period
        
        final_upper_band = [ticker_data.iloc[i]['higher_band'] for i in range(0, start_index)]
        final_lower_band = [ticker_data.iloc[i]['lower_band'] for i in range(0, start_index)]
        bollinger_band = [ticker_data.iloc[i]['higher_band'] for i in range(0, start_index)]
        
        buy_or_sell = [None for i in range(0, start_index)]
        
        for i in range(start_index, len(ticker_data.index)):
            current = i
            previous = i - 1

            current_basic_upper_band = ticker_data.iloc[current]['higher_band']
            current_basic_lower_band = ticker_data.iloc[current]['lower_band']
            current_close = ticker_data.iloc[current]['Close']

            previous_final_upper_band = final_upper_band[-1]
            previous_final_lower_band = final_lower_band[-1]
            previous_bollinger_band = bollinger_band[-1]
            previous_close = ticker_data.iloc[previous]['Close']

            # rules to construct final upper & lower bands
            if current_basic_upper_band < previous_final_upper_band or previous_close > previous_final_upper_band:
                final_upper_band.append(current_basic_upper_band)
            else:
                final_upper_band.append(previous_final_upper_band)

            if current_basic_lower_band > previous_final_lower_band or previous_close < previous_final_lower_band:
                final_lower_band.append(current_basic_lower_band)
            else:
                final_lower_band.append(previous_final_lower_band)

            # calculate the decision based on the bands
            current_final_upper_band = final_upper_band[-1]
            current_final_lower_band = final_lower_band[-1]

            if previous_bollinger_band == previous_final_upper_band and current_close < current_final_upper_band:
                bs = False
                buy_or_sell.append(bs)
                bollinger_band.append(current_final_upper_band)

            elif previous_bollinger_band == previous_final_upper_band and current_close > current_final_upper_band:
                bs = True
                buy_or_sell.append(bs)
                bollinger_band.append(current_final_lower_band)

            elif previous_bollinger_band == previous_final_lower_band and current_close > current_final_lower_band:
                bs = True
                buy_or_sell.append(bs)
                bollinger_band.append(current_final_lower_band)

            elif previous_bollinger_band == previous_final_lower_band and current_close < current_final_lower_band:
                bs = False
                buy_or_sell.append(bs)
                bollinger_band.append(current_final_upper_band)

        ticker_data['bollinger_band'] = bollinger_band
        ticker_data['final_higher_band'] = final_upper_band
        ticker_data['final_lower_band'] = final_lower_band
        ticker_data['buy_or_sell'] = buy_or_sell
        
        ticker_data = ticker_data.iloc[1:]
        return ticker_data


    def chart_bollinger_band(self, bollinger_bands_df):
        """Chart the Bollinger Bands."""

        # get buy/sell lines 
        buys_line = []
        sells_line = []

        for i in range(0, len(bollinger_bands_df.index)):
            value = bollinger_bands_df.iloc[i]['bollinger_band']
            if bollinger_bands_df.iloc[i]['buy_or_sell']:
                buys_line.append(value)
                sells_line.append(None)
            else:
                sells_line.append(value)
                buys_line.append(None)
            
        # extract data for candles
        candlestick = go.Candlestick(
            name='candles',
            x=bollinger_bands_df.index,
            open=bollinger_bands_df['Open'],
            high=bollinger_bands_df['High'],
            low=bollinger_bands_df['Low'],
            close=bollinger_bands_df['Close']
        )

        # extract data for lower & higher bands
        lower_band = go.Scatter(
            name='lower_band',
            x=bollinger_bands_df.index,
            y=bollinger_bands_df['lower_band'],
            line=dict(color="purple"),
        )

        higher_band = go.Scatter(
            name='higher_band',
            x=bollinger_bands_df.index,
            y=bollinger_bands_df['higher_band'],
            line=dict(color="purple"),
        )

        # extract data for final bands
        final_upper_band_line = go.Scatter(
            name='final_upper_band',
            x=bollinger_bands_df.index,
            y=bollinger_bands_df['final_higher_band'],
            line=dict(color='green'),
        )

        final_lower_band_line = go.Scatter(
            name='final_lower_band',
            x=bollinger_bands_df.index,
            y=bollinger_bands_df['final_lower_band'],
            line=dict(color='red'),
        )

        # extract data for bollinger band trend overall line
        bollinger_band_line = go.Scatter(
            name='bollinger_band_line',
            x=bollinger_bands_df.index,
            y=bollinger_bands_df['bollinger_band'],
            line=dict(color='black'),
        )

        buy_line = go.Scatter(
            name='buy_line',
            x=bollinger_bands_df.index,
            y=buys_line,
            line=dict(color='purple'),
        )

        sell_line = go.Scatter(
            name='sell_line',
            x=bollinger_bands_df.index,
            y=sells_line,
            line=dict(color='pink')
        )

        data = [candlestick, final_upper_band_line, final_lower_band_line, bollinger_band_line, buy_line, sell_line]
        fig = go.Figure(
            data=data,
        )
        
        fig.update_layout(
            autosize=False,
            width=1200,
            height=800,)

        fig.show()


    def simulate_portfolio_on_strategy(self, bollinger_bands_df):
        """Simulate the portfolio holdings based on the strategy."""

        # initialization
        portfolio_vals = [self.initial_amount]
        percent_differences = [0.0]
        buying = False

        initial_amt = self.initial_amount
        
        for i in range(1, len(bollinger_bands_df.index)):
            current = i
            previous = i - 1
            current_trade = bollinger_bands_df.iloc[current]['buy_or_sell']
            previous_trade = bollinger_bands_df.iloc[previous]['buy_or_sell']
            raw_percent_difference = 0
            
            # buy order
            if previous_trade == False and current_trade == True:
                buying = True
                
            # sell order
            elif previous_trade == True and current_trade == False:
                previous_price = bollinger_bands_df.iloc[previous]['Close']
                current_price = bollinger_bands_df.iloc[current]['Close']
                raw_percent_difference = ((current_price - previous_price) / previous_price)
                buying = False
                
            if buying:
                previous_price = bollinger_bands_df.iloc[previous]['Close']
                current_price = bollinger_bands_df.iloc[current]['Close']
                raw_percent_difference = ((current_price - previous_price) / previous_price)
                
            percent_difference = 1 + raw_percent_difference
            initial_amt = initial_amt * percent_difference

            percent_differences.append(raw_percent_difference)
            portfolio_vals.append(initial_amt)
                
        return portfolio_vals, percent_differences


    def simulate_portfolio_strategy_with_slippage(self, bollinger_bands_df, buy_slippage=1, sell_slippage=1):
        """Reality isn't clean, simulate the pricing if the prices deviate from what was specified."""

        buying_periods = deque()
        current_day = 1
        previous_day = 0

        portfolio_vals = [self.initial_amount]
        initial_amt = self.initial_amount

        while current_day != len(bollinger_bands_df.index):
            previous_day_row = bollinger_bands_df.iloc[previous_day]
            current_day_row = bollinger_bands_df.iloc[current_day]

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

        while i != len(bollinger_bands_df.index):
            raw_percentage = 0
            if bp:
                if i < bp[0]:
                    raw_percentage = 0
                elif bp[0] <= i <= bp[1]:
                    previous_price = bollinger_bands_df.iloc[i - 1]['Close']
                    current_price = bollinger_bands_df.iloc[i]['Close']
                    raw_percentage = ((current_price - previous_price) / previous_price)
                elif i > bp[0]:
                    bp = buying_periods.popleft() if buying_periods else None

            percentage_difference = 1 + raw_percentage
            initial_amt = initial_amt * percentage_difference

            percentage_differences.append(percentage_difference)
            portfolio_vals.append(initial_amt)

            i += 1

        return portfolio_vals, percentage_differences
        

    def benchmark_strategy(self, bollinger_bands_df, benchmark_ticker='VTI'):
        """Benchmark bollinger band/trend-following against simply holding another `ticker` (default VTI)."""

        start_date = bollinger_bands_df.iloc[0].name
        end_date = bollinger_bands_df.iloc[-1].name

        logging.info(f"Start date: {start_date}")
        logging.info(f"End date: {end_date}")

        # get benchmark data & calculate portfolio change by simply holding
        benchmark_df = yf.Ticker(benchmark_ticker).history(period="max")
        end_date = min(end_date, benchmark_df.iloc[-1].name)
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
            benchmark_percent_differences.append(raw_percent_difference)
            
        benchmark_df['portfolio_values'] = b_portfolio_values
        benchmark_df['percentage_change'] = benchmark_percent_differences

        # get strategy principal & total values
        total_return = bollinger_bands_df.loc[end_date]['portfolio_values']
        b_total_return = benchmark_df.loc[end_date]['portfolio_values']
        
        # get benchmark principal & total values
        principal = benchmark_df.loc[start_date]['portfolio_values']
        b_principal = benchmark_df.loc[start_date]['portfolio_values']

        # cumulative return
        cumulative_return = ((total_return - principal) / principal)

        # get benchmark cumulative return
        b_cumulative_return = ((b_total_return - b_principal) / b_principal)

        # annual return
        annual_return = (1 + sum(bollinger_bands_df['percentage_change'].values) / len(bollinger_bands_df['percentage_change'].values)) ** 252 - 1

        # benchmark annual return
        b_annual_return = (1 + sum(benchmark_df['percentage_change'].values) / len(benchmark_df['percentage_change'].values)) ** 252 - 1

        # annualized volatility
        percent_differences = list(bollinger_bands_df['percentage_change'].values)
        mean = (sum(percent_differences) / len(percent_differences))
        std_dev = (sum([((x - mean) ** 2) for x in percent_differences]) / len(percent_differences)) ** 0.5
        annualized_volatility = std_dev * (252 ** 0.5)  # multiply by sqrt(252) to annualize

        # benchmark annualized volatility
        b_percent_differences = list(benchmark_df['percentage_change'].values)
        b_mean = (sum(b_percent_differences) / len(b_percent_differences))
        b_std_dev = (sum([((x - b_mean) ** 2) for x in b_percent_differences]) / len(b_percent_differences)) ** 0.5
        b_annualized_volatility = b_std_dev * (252 ** 0.5)  # multiply by sqrt(252) to annualize

        # sharpe ratio
        risk_free_rate = 0.005
        sharpe_ratio = (annual_return - risk_free_rate) / annualized_volatility

        # benchmark sharpe ratio
        b_sharpe_ratio = (b_annual_return - risk_free_rate) / b_annualized_volatility

        # creating a Dataframe object from dictionary
        benchmark_comparison_dict = {
            'principal' : [self.initial_amount, self.initial_amount],
            'total_return' : [total_return, b_total_return],
            'cumulative_return' : [cumulative_return, b_cumulative_return],
            'annual_return': [annual_return, b_annual_return],
            'annualized_volatility': [annualized_volatility, b_annualized_volatility],
            'sharpe_ratio': [sharpe_ratio, b_sharpe_ratio],
        }
        benchmark_comparison_df = pd.DataFrame(benchmark_comparison_dict, index = ['strategy', 'benchmark',])

        return benchmark_comparison_df, benchmark_df


    def visualize_benchmark_against_strategy(self, bollinger_bands_df, benchmark_df):
        """Chart the benchmarked strategy against the Bollinger Band based strategy."""

        benchmark_line = go.Scatter(
            name='benchmark_VTI',
            x=benchmark_df.index,
            y=benchmark_df['portfolio_values'],
            line=dict(color='red'),
        )

        strategy_line = go.Scatter(
            name=f'bollinger_band_strategy_{self.ticker}',
            x=bollinger_bands_df.index,
            y=bollinger_bands_df['portfolio_values'],
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


    def get_trade_decision(self, bollinger_bands_df):
        """Helper function to get buy/sell/no-op decision."""
        
        last_trading_day = bollinger_bands_df.iloc[-1]
        second_to_last_trading_day = bollinger_bands_df.iloc[-2]
        
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


    def generate_daily_bollinger_band(self, period="6mo"):
        """Generate the bands for the day."""

        bollinger_bands_df = self.generate_bollinger_bands_for_ticker(period=period)
        portfolio_over_time, percent_differences = self.simulate_portfolio_on_strategy(bollinger_bands_df)

        bollinger_bands_df['portfolio_values'] = portfolio_over_time
        bollinger_bands_df['percentage_change'] = percent_differences

        return bollinger_bands_df


    def execute_daily_trade_decision(self):
        """Used for real(ish) time alerts."""
        
        bollinger_bands_df = self.generate_daily_bollinger_band()
        trade_decision = self.get_trade_decision(bollinger_bands_df)
        logging.info(f"Trade decision: {trade_decision}")

        if not self.debug_mode:
            should_notify = False
            
            # notify if the last date is today & there hasn't been a notification yet
            last_uploaded_row = mongo_instance.read_last_row_from_mongo(self.ticker)
            last_current_row = bollinger_bands_df.iloc[-1]

            last_current_date = last_current_row.name.date()
            today_date = datetime.today().date()

            if last_uploaded_row:
                last_uploaded_date = last_uploaded_row.Date.date()

                # should notify if today's date is not equal to the last uploaded date and today's date equals the last current date from the dataframe
                if today_date != last_uploaded_date and today_date == last_current_date:
                    should_notify = True
            
            # only notify if it is the first upload and today's date is the last date in the dataframe
            elif not last_uploaded_row and last_current_date == today_date:
                import pdb;
                pdb.set_trace()

                should_notify = True

            if should_notify:
                logging.info(f"Attempting to notify for {self.ticker}")

                # send email w/ trade decision
                notified = email_sender.send_email(SENDER_EMAIL, RECEIVER_EMAIL, trade_decision)

                # add the notified boolean to the row
                last_index = bollinger_bands_df.index[-1]
                bollinger_bands_df['notified'] = False
                bollinger_bands_df.loc[last_index, 'notified'] = notified

                # upload data to Mongo
                mongo_instance.write_last_row_to_mongo(self.ticker, bollinger_bands_df)
