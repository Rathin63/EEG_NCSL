"""
String Processing Utilities Module

This module provides utility functions for string manipulation,
particularly for splitting and parsing strings containing mixed
alphanumeric content.

Functions:
    text_num_split: Split strings into text and number components
    split_label: Parse labels into region and number (e.g., for EEG channels)
"""


def text_num_split(item):
    """
    Split a string into text and number components.

    This function finds the first digit in a string and splits the string
    at that point, returning the text portion and the numeric portion as
    separate elements.

    Parameters
    ----------
    item : str
        Input string containing both text and numbers

    Returns
    -------
    list
        A list with two elements: [text_part, number_part]
        Returns None if no digit is found

    Examples
    --------
    ### text_num_split("foobar12345")
    ['foobar', '12345']

    ### text_num_split("channel42")
    ['channel', '42']

    ### text_num_split("A1B2C3")
    ['A', '1B2C3']

    ### text_num_split("nodigits")
    None  # No digits found

    Notes
    -----
    - Splits at the FIRST digit encountered
    - Remaining text after first digit stays with number part
    - Useful for parsing identifiers like "ROI123" or "Channel5"
    """
    for index, letter in enumerate(item, 0):
        if letter.isdigit():
            return [item[:index], item[index:]]


def split_label(s):
    """
    Split an EEG channel label into region and number components.

    This function is specifically designed for parsing EEG channel labels
    in bipolar format. It extracts the alphabetic region identifier and
    the numeric channel number.

    Parameters
    ----------
    s : str
        EEG channel label (e.g., "FP1", "T3", "O2")

    Returns
    -------
    tuple
        A tuple containing:
        - region (str): The alphabetic part of the label
        - number (int): The numeric part of the label

    Examples
    --------
    ### split_label("FP1")
    ('FP', 1)

    ### split_label("T3")
    ('T', 3)

    ### split_label("CZ10")
    ('CZ', 10)

    ### split_label("O2")
    ('O', 2)

    Application
    -----------
    Commonly used in EEG data processing to:
    - Convert monopolar to bipolar montages
    - Sort channels by region and number
    - Group channels by brain region

    Notes
    -----
    - Assumes label ends with digits
    - Converts number to integer type
    - Designed for standard EEG nomenclature (10-20 system)
    """
    # Put the eeg data into bipolar format
    region = s.rstrip('0123456789')
    number = s[len(region):]
    return region, int(number)