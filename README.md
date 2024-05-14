# Main Code Setup Guide

This guide provides step-by-step instructions to set up the environment, modify necessary files, and download the required datasets for the project.


## Step 1: Download the Repository and Setup the Environment

Follow these commands in your terminal to clone the repository and set up the environment:

```bash
git clone git@github.com:JiachenJasonZhong/JAWZ_Big_Data.git
conda install mamba -n base -c conda-forge # Note this using conda 23.11.0                        # Update May 13,2024： Alternatively, you should use the fresh install (by install) with the following link (make sure the path to be added to the path at the environmental variables): https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html
mamba env create -f /path/to/CSE_project5.yaml # Replace /path/to/ with your specific path
conda activate CSE_project5
pip install sklearn-deap # Note this is using pip 24.0 
conda install scikit-learn=0.23.2 -y
pip install -U scikit-multiflow==0.5.3
```

## Step 2: Modify the _data.py file 

Add your specific path to data_config.yaml in the /src/configuration/data/ directory:

For instance, where is says "folder:" in line 5, change that your full path. 

## Step 3: Download the data 

Download the following data sets and place them in the folder as shown in the photo labeled Data Setup (you have to make a separate elliptic file and place the various elliptic files in there)

A note about the elliptic_embs.csv, that can be found in the following google drive link: 
https://drive.google.com/drive/folders/1xxJgmMPKVGLymI90fX1JxHFU9GCEJvK-

The other elliptic dataset can be found here: 

https://www.kaggle.com/ellipticco/elliptic-data-set

Please download the Complete.csv found in Account_Stats folder the following Github

https://github.com/sfarrugia15/Ethereum_Fraud_Detection

The weather data can be found here:

http://users.rowan.edu/~polikar/res/nse/weather_data.zip
