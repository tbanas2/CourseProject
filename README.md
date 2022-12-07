# tBanaszak - tbanas2 - CS410 Final Project

## Repository Contents
The repository contains code modules, data files and project submission materials. The following files are the ones needed for reviewer grading

### !FINAL_CODE.ipynb
This is the only code file that needs to be accessed for review. This is the Jupyter notebook to run that applies my project's system for demonstration; in-line documentation is included here in addition to the separate report

### !CS410_Final_Documentation_Report.pdf
External documentation and project summary report

### Documentation and Project Video 
Available through the following Box link
https://uofi.box.com/s/gw62xkaaw0davyugcc6x8szpp3yut2ch

# IMPORTANT - NOTES ON RUNNING NOTEBOOK - PLEASE READ COMPLETELY
It's recommended that the system be run by creating a new virtual environment and using the repository's requirements.txt file to install the necessary libraries.

Anaconda is the easiest way to do this. For example, the steps would be:

clone the repository to a local directory

Create the environment with
conda create --name [ENVIRONMENT NAME]

Activate the environment
conda activate [ENVIRONMENT NAME]

Then, from within the cloned repo
pip install -r requirements.txt 

## THE PROJECT RELIES ON LARGE DEPENDENCIES 
The project uses Tensorflow which is nearly 450MB. Downloading this will take a while.

You'll also need Jupyter notebook installed.

Lastly, be aware that the notebook contains an option to re-train a neural network model or use the pre-trained one. The default behavior, which is recommended, is to just use the pre-trained one.




