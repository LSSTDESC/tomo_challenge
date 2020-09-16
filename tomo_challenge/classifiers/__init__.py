import os
import sys
from .base import Tomographer

def all_python_files():
    root_dir = os.path.dirname(__file__)
    names = []
    for filename in os.listdir(root_dir):
        if filename.endswith('.py') and filename != '__init__.py':
            name = filename[:-3]
            names.append(name)
    return names

for name in all_python_files():
    try:
        __import__(name, globals(), locals(), level=1)
    except Exception as error:
        sys.stderr.write(f"Failed to import {name}: '{error}'")
