FROM nvidia/cuda:11.2.2-devel-ubuntu16.04

ENV USERNAME=developer
ENV DEBIAN_FRONTEND=noninteractive

COPY installer.sh /installer.sh 
RUN sh installer.sh

ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

ENTRYPOINT ["/bin/sh"]
