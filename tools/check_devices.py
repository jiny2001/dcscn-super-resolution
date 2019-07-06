"""
Script to check the list of available devices on current host
"""

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())