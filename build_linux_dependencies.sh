#!/bin/sh

cd xgboost
git submodule init
git submodule update

make -j4
cd python-package
python setup.py install

if [ $? -ne 0 ] ; then
    cd ../..
    exit 1
fi

cd ../..

cd alphamind/pfopt
./build_linux.sh
if [ $? -ne 0 ] ; then
    cd ../..
    exit 1
fi

cd ../..