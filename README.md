## Semi-Automated Trading with Bollinger Bands
Module built to trade and research different flavors of [Bollinger Bands](https://en.wikipedia.org/wiki/Bollinger_Bands) for desired assets.


## Installation
- python >= 3.6
- `pip install -r requirements3.txt`

## Architecture
The main code lives in `src/super_trend_lib.py`, with some helper classes in `utils.py`.
Within `super_trend_lib.py`, there's a few functions of note that are described further below (and in the comments).

Within `utils.py`, there's two helper classes `MongoInstance` and `EmailSender`. `MongoInstance` wraps functionality to read and write dataframes for given tickers. `EmailSender` is a wrapper class to send emails from desired sender to desired recipient.

There's another architectural component here related to deployment, hence the inclusion of a `Dockerfile` and `docker-compose.yml`. More on that later.

## Module Details
Not every function is described here, just the ones worthy of further explanation.

<b>`init`</b>

This creates a SuperTrendRunner (obviously), but it's worth discussing the parameters.

- `ticker` - this is the desired asset to evaluate (e.g. `NVDA`, `TSLA`)
- `multiplier` - [from Wikipedia](https://en.wikipedia.org/wiki/Bollinger_Bands#:~:text=Bollinger%20Bands%20consist,and%202%2C%20respectively.), this is the multiplier for the standard deviation or average true range over a certain period
- `rolling_period` - rolling number of days to take the standard deviation or average true range
- `initial_amount` - initial portfolio amount
- `debug_mode` - flag to control whether to write to Mongo & send an email (set this to `True` for research-only activities)

<br />

<b>`explore(period="6mo)`</b>

- This creates the super_trend dataframe for a given lookback period (defaults to 6mo)
- Charts the super_trend along with key bands + buy/sell lines
- Simulates the portfolio earnings using this strategy
- Benchmarks the strategy against holding a default ticker (normally VTI)
- Visualizes portfolio earnings for the strategy & the benchmark

Here's an example.

<br />

<b>`generate_super_trend_for_ticker`</b>

This function is the main driver for generating all the bands. Won't go too in-depth here as the code is the best documentation.

<br />

<b>`simulate_portfolio_on_strategy`</b>

Using the buy/sell points from the previous function, this calculates the percent changes day-to-day and the consequent portfolio values. The percent changes are used downstream to calculate volatility while the portfolio values are eventually charted in another function.

<br />

<b>`benchmark_strategy`</b>

This benchmarks the super trend against simply holding another asset (defaults to `VTI`). Key metrics include:

- `cumulative_return` - percentage total return from principal to total
- `annual_return` - annualized return as a percentage (this is a bogus value if the lookback period is < 1 year)
- `volatility` - standard deviation of percent differences
- `sharpe_ratio` - performance relative to a risk-free asset (this metric could definitely use some tuning)
- `calmar_ratio` - average annual rate of return divided by the max drawdown (this is currently implemented poorly as is because it technically is supposed to be calculated every month) 

<br />

<b>`execute_daily_trade_decision`</b>

This is the function that is intended to be run every day after market close for a given asset and then indicate a `BUY`, `SELL`, or `NO-OP` decision. Once the decision is calculated, the resulting dataframe (or last row) is uploaded to Mongo and an email with the decision is sent to the desired end user.

## Deployment

There's a research component and a production component to this project. For simple research purposes, it's enough to just clone the repo, install the dependencies, and fire up a notebook. A production environment that monitors the buy/sell lines, notifies users, stores data, and regenerates charts all on a daily cadence is a slightly different flavor of problem than research.

Rather than paying for compute & storage, self-hosting was an appealing alternative. For a bespoke self-hosted rig, the following was desirable:
- "Always-on"
- Globally accessible
- Easily reproducible in case of failure
- Able to serve as both research & production environment
- Capable of interacting with a database that persists through application lifetimes

Containerization naturally satisfied these requirements. The `Dockerfile` and `docker-compose.yml` builds two containers:

- A MongoDB container to serve as a persistent data store for the tickers of interest
- A [VSCode server](https://github.com/coder/code-server) to serve as a web-based IDE to support both research & deployment of daily asset monitoring

Breaking down the `Dockerfile` a bit further, there's some important things to note:

- A virtual environment is in the container based on the `requirements3.txt`
- The Python & Jupyter extensions are installed into the VSCode container
- [Tailscale](https://tailscale.com/) is installed to expose this container to a user's Tailnet (can also be publicly exposed with [Tailscale Funnel](https://tailscale.com/kb/1223/funnel))
- `src` & `research` directories are mounted as a volume into the container, meaning that edits across the web-based IDE occur locally (this is one of the major perks of this setup!)
- Similarly, the `mongo-data` folder is mapped from the host to `/data/db` inside the container, ensuring MongoDB's data persists across container restarts
- Important environment variables are stored in a `.env` file and ingested at container build time

To build the containers, run `docker-compose up` and to tear them down, run `docker-compose down`.

## Areas for Improvement
In no particular order:

- The code is currently structured to only work with a lookback period, but a custom time range would be useful, especially for backtesting in certain conditions.
- I've been unable to figure out how to use Tailscale in an automated way while building the container. I have to enter my container, run `blah`, and then `tailscale up --authkey [blah]` to actually add the container into my Tailnet. Similarly, I have to do `tailscale funnel` to then expose the container to the broader Internet.
- The code is almost certainly riddled with a few small bugs. Unit tests would give greater confidence in this implementation.
- Typing is very useful, but currently only used in `utils.py`. It should be added to `super_trend_lib.py`.
- Much of the code right now is very strongly tied to the `yfinance` API, a better approach would be adding a middle layer of abstraction to decouple the business logic from the data provider.
- There's many flavors of Bollinger Bands to experiment with and a rigorous meta-analysis evaluating different combinations of the paramemters (lookback, asset classes, time ranges, markets, contrarian strategies) would be an interesting exercise.

