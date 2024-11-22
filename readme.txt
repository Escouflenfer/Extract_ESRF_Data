Python code and Jupyter Notebook provided by Institut Néel under the MIAM Project (PEPR DIADEM).
Package used to vizualise and extract data from .h5 high-throughtput XRD data at BM02 (ESRF).
You can contact me for any issues at william.rigaut@neel.cnrs.for

version: 1.0
author: williamrigaut

Getting started:
    You will need a recent version of python (3.8 or higher) in order to run the python code
    Installing Jupyter Notebook is also highly recommanded since a detail tutorial is provided.

    Then you will need to create a new python environnement to import the required libraries,
    you can do that with the following command in a terminal:
        python3 -m venv .venv
    and then:
        source .venv/bin/activate
    Finally to import all the libraries:
        pip install -r requirements.txt

Once the installation is done, you can open the Notebook Extract_ESRF-Data.ipynb
Since datafile sizes are huge for ESRF (bm02), no example dataset is provided, contact me if you need an example. 
Please contact me if you need assistance or if you think that something is not working as intended.