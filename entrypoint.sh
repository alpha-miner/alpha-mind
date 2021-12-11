#!/bin/sh

export PYTHONPATH=$PYTHONPATH:/
export DB_VENDOR="mysql"
# export DB_URI="mysql+mysqldb://dxrw:dxRW20_2@121.37.138.1:13317/dxtest?charset=utf8"
export DB_URI="mysql+mysqldb://reader:Reader#2020@121.37.138.1:13316/vision_product?charset=utf8"
export FACTOR_TABLES="factor_momentum"
jupyter lab --ip="0.0.0.0" --port=8080 --allow-root --ServerApp.token='' --ServerApp.password='sha1:f7761f682bc4:1aba35e73699fe62570573de373bf95b491022a7'