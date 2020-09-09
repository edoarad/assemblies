import gc
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def pytest_runtest_teardown(item, nextitem):
    gc.collect()