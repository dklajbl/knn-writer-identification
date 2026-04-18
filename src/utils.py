import os

from src.env_vars import INTERN_AUTHORS_FILE_PATH

def convert_sec_to_hours_minutes_seconds(sec):
    # if sec is double, convert to int
    if isinstance(sec, float):
        sec = int(sec)
    hours = sec // 3600
    minutes = (sec % 3600) // 60
    seconds = sec % 60
    # if hours is 0, return minutes and seconds only
    ret_str = ""
    if hours > 0:
        ret_str += f"{hours}h "
    if minutes > 0:
        ret_str += f"{minutes}m "
    if seconds > 0:
        ret_str += f"{seconds}s"
    return ret_str.strip()
