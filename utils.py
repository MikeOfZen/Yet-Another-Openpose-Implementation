import config as c
import datetime

def now():
    return c.RUN_NAME+datetime.datetime.now().strftime("%d%a%m%y-%H%M")