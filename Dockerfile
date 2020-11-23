FROM wegamekinglc/python:3.7-slim-stretch-aliyun

LABEL maintainer = "scrappedprince.li@gmail.com"
RUN apt-get update && apt-get install git cmake build-essential gfortran default-libmysqlclient-dev -y
COPY ./alphamind /alphamind
COPY ./notebooks /notebooks

WORKDIR /alphamind/pfopt
RUN export BUILD_TEST=OFF
RUN export REDIRECT=$1
RUN bash ./build_linux.sh

WORKDIR /
COPY ./requirements.txt /requirements.txt
RUN pip install numpy==1.19.1 -i https://pypi.douban.com/simple
RUN pip install -r /requirements.txt -i https://pypi.douban.com/simple
RUN pip install finance-python>=0.8.1 -i https://pypi.douban.com/simple

COPY ./setup.py /setup.py
COPY ./setup.cfg /setup.cfg
RUN python setup.py build_ext --inplace

EXPOSE 8080
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /notebooks
ENTRYPOINT ["/entrypoint.sh"]
CMD []