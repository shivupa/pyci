#!/bin/bash
export PYTHONPATH=/Users/shiv/Documents:$PYTHONPATH
export DYLD_LIBRARY_PATH=/Users/shiv/Documents/pyscf/lib/deps/lib:$DYLD_LIBRARY_PATH
source activate python3
python asci.py
