#!/bin/sh

cd alphamind/pfopt

./build_linux.sh

if [ $? -ne 0 ] ; then
    cd ../..
    exit 1
fi

cd ../..