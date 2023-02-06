## Bollinger Bands



## Installation
- python >= 3.6
- `pip install -r requirements3.txt`

## Architecture
A dataframe for an asset is first created locally, then uploaded to Mongo.
A cron runs daily and generates a decision [No-Op, Buy, Sell] & sends it via email.

<b>Instantiating</b>:

<b>Trading</b>:
```
from core.super_trend_lib import SuperTrendRunner

# JSON_PATH = '/Users/vaibhav/projects/super_trend'
JSON_PATH = '/root/projects/supertrend'

vgt_runner = SuperTrendRunner(ticker='VGT', json_path=JSON_PATH, debug_mode=False)
vgt_runner.execute_daily_trade_decision()
```

## Deployment
Module is agnostic to local or hosted setup. 

