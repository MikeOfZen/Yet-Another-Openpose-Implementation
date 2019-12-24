import datetime


def now():
    return datetime.datetime.now().strftime("%d%a%m%y-%H%M")
