from super_trend_lib import SuperTrendRunner

# example of how to run supertrend on given assets, post-initialization
runners = [
    SuperTrendRunner(ticker='PLTR', debug_mode=False),
    SuperTrendRunner(ticker='VGT', debug_mode=False),
    SuperTrendRunner(ticker='VTI', debug_mode=False),
    SuperTrendRunner(ticker='LAND', debug_mode=False),
]

for runner in runners:
    runner.execute_daily_trade_decision()
