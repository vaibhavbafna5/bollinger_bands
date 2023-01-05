from super_trend_lib import SuperTrendRunner

# JSON_PATH = '/Users/vaibhav/projects/super_trend'
JSON_PATH = '/root/projects/supertrend'

land_runner = SuperTrendRunner(ticker='LAND', json_path=JSON_PATH, debug_mode=False)
land_runner.execute_daily_trade_decision()