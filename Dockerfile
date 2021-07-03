FROM continuumio/anaconda3:2021.05

LABEL maintainer = "scrappedprince.li@gmail.com"
RUN apt-get update && apt-get install build-essential default-libmysqlclient-dev coinor-cbc coinor-libcbc-dev -y
ENV COIN_INSTALL_DIR /usr

WORKDIR /
COPY ./requirements_docker.txt /requirements.txt
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