## Semi-Automated Trading with Bollinger Bands
Module built to trade on customized [Bollinger Bands](https://en.wikipedia.org/wiki/Bollinger_Bands) for desired assets.


## Installation
- python >= 3.6
- `pip install -r requirements3.txt`

## Architecture
A dataframe for an asset is first created locally, then uploaded to Mongo (initialization).
A cron runs daily and generates a decision [No-Op, Buy, Sell] & sends the decision via email (trading).

<b>Instantiating</b>:
```
from super_trend_lib import SuperTrendRunner

vgt_runner = SuperTrendRunner(ticker='VGT', debug_mode=False)
super_trend_df = vgt_runner.initialize_dataframe()  # generate initial dataframe
vgt_runner.write_initial_dataframe_to_mongo(super_trend_df)  # upload dataframe to Mongo

```

<b>Trading</b>:
```
from super_trend_lib import SuperTrendRunner

vgt_runner = SuperTrendRunner(ticker='VGT', debug_mode=False)
vgt_runner.execute_daily_trade_decision()
```

## Deployment
Module is agnostic to local or hosted setup. 

