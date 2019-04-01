#!/bin/sh

cd xgboost
git submodule init
git submodule update

mkdir build
cd build
cmake ..
make -j4
cd ..

cd python-package
python setup.py install

if [ $? -ne 0 ] ; then
    cd ../..
    exit 1
fi

cd ../..

cd alphamind/pfopt

export BUILD_TEST=OFF
export REDIRECT=$1
bash build_linux.sh

if [ $? -ne 0 ] ; then
    cd ../..
    exit 1
fi

cd ../..
