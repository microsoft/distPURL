#!/bin/bash
#Linux System/Subsystem Instalations 
sudo apt-get update
#sudo apt-get upgrade
sudo apt install build-essential 
sudo apt-get install git
sudo apt-get install python3
sudo apt-get install coinor-cbc
sudo apt-get install python3-pip

# create the install directory under $HOME
cd ~
mkdir install_dir
cd install_dir

#Python Packages Installs
pip3 install pickle-mixin
pip3 install pathlib
pip3 install typing 
pip3 install numpy
pip3 install pandas
pip3 install tqdm
pip3 install sklearn
pip3 install click
pip3 install pytz
pip3 install Datetime
pip3 install matplotlib
pip3 install colorama
pip3 install joblib
pip3 install pulp
pip3 install mip
pip3 install scipy
pip3 install seaborn
python3 -m pip install -i https://pypi.gurobi.com gurobipy