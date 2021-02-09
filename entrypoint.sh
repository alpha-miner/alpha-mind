#!/bin/sh

export PYTHONPATH=$PYTHONPATH:/
export DB_VENDOR="mysql"
# export DB_URI="mysql+mysqldb://dxrw:dxRW20_2@121.37.138.1:13317/dxtest?charset=utf8"
export DB_URI="mysql+mysqldb://reader:Reader#2020@121.37.138.1:13316/vision_product?charset=utf8"
export FACTOR_TABLES="factor_momentum"
jupyter lab --ip="0.0.0.0" --port=8080 --allow-root --NotebookApp.token='' --NotebookApp.password=''