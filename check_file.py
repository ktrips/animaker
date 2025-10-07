import os
import pathlib
import datetime
import time
import platform

p = pathlib.Path('test_api.py')

print(p.stat())
print(p.stat().st_mtime)
print(datetime.datetime.fromtimestamp(p.stat().st_mtime))

