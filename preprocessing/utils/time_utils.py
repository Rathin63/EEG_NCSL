"""
Time Conversion Utilities Module

This module provides utility functions for converting between different time formats,
including seconds, decimal hours, and HH:MM:SS format. Also includes date computation
utilities for handling event times.

Functions:
    sec2h: Convert seconds to decimal hours
    sec2time: Convert seconds to HH:MM:SS format
    time2sec: Convert HH:MM:SS format to seconds
    time_sum: Add two times in HH:MM:SS format
    compute_date: Determine date based on time comparisons

Dependencies:
    numpy
"""

import numpy as np


def sec2h(t):
    """
    Convert seconds to decimal hours.

    This function takes a time value in seconds and converts it to
    decimal hours, where the fractional part represents minutes and
    seconds as decimal portions of an hour.

    Parameters
    ----------
    t : int or float
        Time in seconds (valid range: 0 to 86400 for 24 hours)

    Returns
    -------
    float
        Time in decimal hours (e.g., 1.5 = 1 hour 30 minutes)

    Examples
    --------
    ### sec2h(3600)  # 1 hour
    1.0

    ### sec2h(5400)  # 1 hour 30 minutes
    1.5

    ### sec2h(3661)  # 1 hour 1 minute 1 second
    1.0169444444444444

    Notes
    -----
    - Useful for calculations requiring time in decimal format
    - Precision depends on input precision
    """
    hours = np.floor(t / 3600)
    t = t - hours * 3600
    mins = np.floor(t / 60)
    secs = t - mins * 60
    h = hours + (mins / 60) + (secs / 3600)
    return h


def sec2time(t):
    """
    Convert seconds to HH:MM:SS format string.

    This function takes a time value in seconds and converts it to
    a readable time string in HH:MM:SS format.

    Parameters
    ----------
    t : int or float
        Time in seconds (valid range: 0 to 86400 for 24 hours)

    Returns
    -------
    str
        Time in HH:MM:SS format (e.g., "1:30:45")

    Examples
    --------
    ### sec2time(3600)
    "1:0:0"

    ### sec2time(5445)
    "1:30:45"

    ### sec2time(86399)
    "23:59:59"

    Notes
    -----
    - Hours are not zero-padded (1:30:45 not 01:30:45)
    - Useful for displaying time in human-readable format
    """
    hours = np.floor(t / 3600)
    t = t - hours * 3600
    mins = np.floor(t / 60)
    secs = t - mins * 60
    time = f"{int(hours)}:{int(mins)}:{int(secs)}"
    return time


def time2sec(hour):
    """
    Convert HH:MM:SS format to seconds.

    This function parses a time string in HH:MM:SS format and
    converts it to the total number of seconds.

    Parameters
    ----------
    hour : str
        Time in HH:MM:SS format (e.g., "1:30:45")

    Returns
    -------
    int
        Total time in seconds

    Examples
    --------
    ### time2sec("1:0:0")
    3600

    ### time2sec("1:30:45")
    5445

    ### time2sec("23:59:59")
    86399

    Notes
    -----
    - Expects exactly three components separated by colons
    - Components can be floats but will be converted to integers
    - No validation for valid time ranges
    """
    time = hour.split(':')
    h = time[0]
    m = time[1]
    s = time[2]
    h = int(float(h))
    m = int(float(m))
    s = int(float(s))
    t = h * 60 * 60 + m * 60 + s
    return t


def time_sum(time1, time2):
    """
    Add two times in HH:MM:SS format.

    This function takes two time strings in HH:MM:SS format,
    adds them together, and returns the sum in the same format.
    Hours wrap around at 24 (modulo 24).

    Parameters
    ----------
    time1 : str
        First time in HH:MM:SS format
    time2 : str
        Second time in HH:MM:SS format

    Returns
    -------
    str
        Sum of time1 and time2 in HH:MM:SS format
        Hours are taken modulo 24

    Examples
    --------
    ### time_sum("1:30:45", "2:45:30")
    "4:16:15"

    ### time_sum("23:30:00", "1:00:00")
    "0:30:0"  # Wraps around 24 hours

    ### time_sum("12:45:30", "12:45:30")
    "1:31:0"  # 25:31:00 -> 1:31:00

    Notes
    -----
    - Hours wrap around at 24 (uses modulo 24)
    - Useful for calculating durations that may span days
    """
    seconds1 = time2sec(time1)
    seconds2 = time2sec(time2)
    seconds = seconds1 + seconds2

    hours = np.floor(seconds / 3600)
    seconds = seconds - hours * 3600
    mins = np.floor(seconds / 60)
    secs = seconds - mins * 60

    hours = np.mod(hours, 24)

    time = f"{int(hours)}:{int(mins)}:{int(secs)}"
    return time


def compute_date(date="7/26/2023 17:39:32 7/27/2023 5:39:32", event_time="2:15:20"):
    """
    Determine date based on given input dates and event time.

    This function determines whether an event occurred on the start date
    or end date by comparing the event time with the start time. It assumes
    the date range spans at most two consecutive days.

    Parameters
    ----------
    date : str
        A string containing start and end date/times in the format:
        "MM/DD/YYYY HH:MM:SS MM/DD/YYYY HH:MM:SS"
        Default: "7/26/2023 17:39:32 7/27/2023 5:39:32"
    event_time : str
        The time of the event in HH:MM:SS format
        Default: "2:15:20"

    Returns
    -------
    str
        The date (MM/DD/YYYY) when the event occurred

    Examples
    --------
    ### compute_date("6/26/2023 17:47:35 6/27/2023 5:47:35", "2:15:20")
    "6/27/2023"  # Event at 2:15 AM is after midnight, so next day

    ### compute_date("6/26/2023 5:47:35 6/26/2023 17:47:35", "12:00:00")
    "6/26/2023"  # Event at noon is same day

    Logic
    -----
    - If start_time_hour > event_time_hour: Event is on the next day
    - If start_time_hour < event_time_hour: Event is on the same day

    Notes
    -----
    - Only compares hours, not full time
    - Assumes the date range represents an overnight period
    - Does not handle equal hours case explicitly
    """
    start_date, start_time, end_date, end_time = str(date).split()
    ev_time_h = int(event_time.split(":")[0])
    start_time_h = int(start_time.split(":")[0])

    if start_time_h < ev_time_h:
        # Same day
        date = start_date

    if start_time_h > ev_time_h:
        # Next day
        date = end_date

    return date