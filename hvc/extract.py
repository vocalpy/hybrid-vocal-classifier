"""
feature extraction
"""

#from standard library
import sys
import glob

#from hvc
from parseconfig import parse

# get command line arguments
args = sys.argv
config_file = args[1]
config = parse(config_file)

if 'extract' not in config:
    raise KeyError('')
else:
    extract = config['extract']

format = feature_extraction['format']
if format=='evtaf':
    import hvc.evfuncs
elif format=='koumura':
    import hvc.koumura
jobs = feature_extraction['jobs']
for job in jobs:
    dirs = job['dirs']
    for dir in dirs:
        os.chdir(dir)
        #run feature extraction script

# save yaml file for select.py with output_dir in it!!!