import pandas as pd
import numpy as np
import datetime
import time

BASE_DATE = '2019-01-01 00:00:00.000'


def convert_to_datetime(x):
    if pd.isna(x):
        return pd.NA
    else:
        return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")


def convert_to_timestamp(x):
    if pd.isna(x):
        return pd.NA
    else:
        return time.mktime(
            datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f").timetuple()
        )


def diff_in_hours(ts1, ts2):
    if pd.isna(ts1) or pd.isna(ts2):
        return pd.NA
    return (convert_to_timestamp(ts1) - convert_to_timestamp(ts2)) / 3600


def index_to_ts(index, time_granularity=0.25, base_date=BASE_DATE):
    base_ts = convert_to_datetime(base_date)
    hours_of_index = pd.Timedelta(hours=index * time_granularity)
    return base_ts + hours_of_index


def ts_to_index(str_ts, time_granularity=0.25, base_date=BASE_DATE):
    """
    Convert a string time into an index for the quarter hour
    """
    if pd.isna(str_ts):
        return pd.NA
    base_ts = convert_to_datetime(base_date)
    ts = convert_to_datetime(str_ts)
    diff_granularity = (ts -
                        base_ts).total_seconds() / (3600 * time_granularity)
    # round to the granularity:
    ts_index = int(diff_granularity)
    # assert that the reversed result is correct
    assert (
        ts - index_to_ts(
            ts_index, time_granularity=time_granularity, base_date=base_date
        )
    ).total_seconds() / 60 < time_granularity * 60
    return ts_index