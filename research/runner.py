"""Testing out how """
from datetime import datetime, timedelta, timezone

import sys
import os
import asyncio
import pytz

# need to do these shenanigans since our "package" is technically stored in another folder
module_path = os.path.abspath(os.path.join('..', 'src'))
if module_path not in sys.path:
    sys.path.append(module_path)

from bollinger_bands import BollingerBands


vgt_bollinger_bands = BollingerBands(ticker="VGT", debug_mode=True)

def seconds_until_target(hour=17, minute=0, tz=pytz.timezone('US/Eastern')):
    """Calculate the number of seconds until the next occurrence of the target time."""

    now = datetime.now(tz)
    target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target_time <= now:
        target_time += timedelta(days=1)
    return (target_time - now).total_seconds()

async def my_async_task():
    """Dummy async task to instantiate and play with Bollinger Bands object."""

    print(f"Task started at {datetime.now(pytz.timezone('US/Eastern'))}")
    print("Running dummy_runner")
    
    vgt_bollinger_bands.explore()

    import pdb;
    pdb.set_trace()

async def run_at_specific_time():
    while True:
        delay = seconds_until_target(hour=17, minute=0)
        print(f"Waiting {delay} seconds until the next run.")

        await asyncio.sleep(delay)
        await my_async_task()

        # optionally, wait a bit before calculating the next run to avoid tight looping around the target time
        await asyncio.sleep(10)

asyncio.run(run_at_specific_time())