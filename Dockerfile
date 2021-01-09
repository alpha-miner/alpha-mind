FROM wegamekinglc/python:3.7-slim-stretch-aliyun

LABEL maintainer = "scrappedprince.li@gmail.com"
RUN apt-get update && apt-get install git cmake build-essential gfortran default-libmysqlclient-dev -y

WORKDIR /
COPY ./requirements.txt /requirements.txt
RUN pip install numpy==1.19.1 -i https://pypi.douban.com/simple
RUN pip install -r /requirements.txt -i https://pypi.douban.com/simple
RUN pip install finance-python>=0.8.1 -i https://pypi.douban.com/simple

WORKDIR /
COPY ./alphamind /alphamind
COPY ./notebooks /notebooks

COPY ./setup.py /setup.py
COPY ./setup.cfg /setup.cfg

EXPOSE 8080
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /notebooks
ENTRYPOINT ["/entrypoint.sh"]
CMD []