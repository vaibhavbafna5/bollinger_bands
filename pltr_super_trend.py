from super_trend_lib import SuperTrendRunner

# JSON_PATH = '/Users/vaibhav/projects/super_trend'
JSON_PATH = '/root/projects/supertrend'

pltr_runner = SuperTrendRunner(ticker='PLTR', json_path=JSON_PATH)
pltr_runner.run()