from super_trend_lib import SuperTrendRunner

# JSON_PATH = '/Users/vaibhav/projects/super_trend'
JSON_PATH = '/root/projects/supertrend'

vti_runner = SuperTrendRunner(ticker='VTI', json_path=JSON_PATH, debug_mode=False)
vti_runner.execute_daily_trade_decision()