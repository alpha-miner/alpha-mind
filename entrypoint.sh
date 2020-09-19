#!/bin/sh

export PYTHONPATH=$PYTHONPATH:/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/alphamind/pfopt/lib
export DB_URI="postgresql+psycopg2://alpha:alpha@localhost:5432/alpha"
jupyter lab --ip="0.0.0.0" --port=8080 --allow-root --NotebookApp.token='' --NotebookApp.password=''