#!/bin/sh

export PYTHONPATH=$PYTHONPATH:/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/alphamind/pfopt/lib
export DB_VENDOR="rl"
export DB_URI="mysql+mysqldb://dxrw:dxRW20_2@121.37.138.1:13317/dxtest?charset=utf8"
jupyter lab --ip="0.0.0.0" --port=8080 --allow-root --NotebookApp.token='' --NotebookApp.password=''