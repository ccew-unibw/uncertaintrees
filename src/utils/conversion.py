"""
Mostly simple helper functions for conversions between data index IDs and real-life formats.
"""

import datetime
import math
from functools import cache

import pendulum
from pendulum.datetime import DateTime
from pendulum.date import Date


@cache
def pgid_to_latlon(id: int) -> tuple[float, float]:
    """converts priogrid cell id to lat/lon coordinates of grid cell center"""
    id -= 1
    lat = math.floor(id / 720) / 2 - 89.75
    lon = id % 720 / 2 - 179.75
    return lat, lon


@cache
def latlon_to_pgid(lat: float, lon: float) -> int:
    """converts lat/lon coordinates of grid cell center to priogrid cell id"""
    row = (lat + 90.25) * 2
    col = (lon + 180.25) * 2
    pgid = (row - 1) * 720 + col
    return int(pgid)


@cache
def get_date(month_id: int) -> DateTime:
    """converts UCDP's month_id to (pendulum) datetime"""
    year = 1980 + math.floor((month_id - 1) / 12)
    month = month_id % 12 if month_id % 12 != 0 else 12
    return pendulum.datetime(year, month, 1)


@cache
def get_month_id(date: datetime.date | datetime.datetime | Date | DateTime) -> int:
    """converts date tp UCDP's month_id"""
    return (date.year - 1980) * 12 + date.month
