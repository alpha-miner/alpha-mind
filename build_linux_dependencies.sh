#!/bin/sh

cd alphamind/pfopt

git submodule init
git submodule update

export BUILD_TEST=OFF
export REDIRECT=$1
bash build_linux.sh

if [ $? -ne 0 ] ; then
    cd ../..
    exit 1
fi

cd ../..
