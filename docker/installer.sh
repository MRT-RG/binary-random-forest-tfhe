#!/bin/sh

apt-get update
apt-get upgrade -y
apt-get install -y apt-utils locales sudo

locale-gen en_US.UTF-8  

# Install apt packages
apt-get install -y build-essential libopencv-dev curl
apt-get install -y python3 python3-dev python3-distutils-extra libboost-python-dev

# Install pip3
curl https://bootstrap.pypa.io/pip/3.5/get-pip.py -o get-pip.py
python3 get-pip.py

# Install pip packages
python3 -m pip install opencv-python numpy scikit-learn opencv-python termcolor

# sudo without password
echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/${USERNAME}

apt-get clean
rm -rf /var/lib/apt/lists/*

# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
