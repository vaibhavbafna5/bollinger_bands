from super_trend_lib import SuperTrendRunner

# JSON_PATH = '/Users/vaibhav/projects/super_trend'
JSON_PATH = '/root/projects/supertrend'

vgt_runner = SuperTrendRunner(ticker='VGT', json_path=JSON_PATH)
vgt_runner.run()