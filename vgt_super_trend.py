from super_trend_lib import SuperTrendRunner

# JSON_PATH = '/Users/vaibhav/projects/super_trend'
JSON_PATH = '/root/projects/supertrend'

vgt_runner = SuperTrendRunner(ticker='VGT', json_path=JSON_PATH, debug_mode=False)
vgt_runner.execute_daily_trade_decision()