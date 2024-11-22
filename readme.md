# Extract ESRF Data

[William Rigaut](https://github.com/escouflenfer)<sup>1, 2</sup>

<sup>1</sup> *Institut Néel, Centre National de la Recherche Scientifique, 38000 Grenoble, France*  
<sup>2</sup> *Université Grenoble Alpes, 38000 Grenoble, France*  


## About

Extract ESRF Data is coded in Python code with a Jupyter Notebook provided by Institut Néel under the MIAM Project (PEPR DIADEM).
Package used to vizualise and extract data from .h5 high-throughtput XRD data at BM02 (ESRF).
You can contact me for any issues at william.rigaut@neel.cnrs.for

Available on Windows, MacOS, and Linux. Requires Python 3.8+.


## Getting Started

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


## Support

If you require support, have questions, want to report a bug, or want to suggest an improvement, please contact me at william.rigaut@neel.cnrs.fr


## License

MIT License