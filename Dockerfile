FROM wegamekinglc/python:3.7-slim-stretch-aliyun

LABEL maintainer = "scrappedprince.li@gmail.com"

RUN apt-get update && apt-get install git cmake build-essential gfortran -y

COPY ./alphamind /alphamind
COPY ./notebooks /notebooks

RUN cd /alphamind/pfopt
RUN export BUILD_TEST=OFF
RUN export REDIRECT=$1
RUN bash build_linux.sh