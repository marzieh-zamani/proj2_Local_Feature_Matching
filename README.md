# ENSE 885AY assignment 2: [Local Feature Matching]


# Project Review
- Project subject: Local Feature Matching

- Project objectives: Creating a local feature matching algorithm using following steps:

Steps to local feature matching between two images (image1 & image 2):
1.	Detecting interest point in both images using Harris detector followed by adaptive non-maximal suppression (ANMS);
2.	Describing local features of interest points in both images using a SIFT-like algorithm;
3.	Comparing and matching interest points in both images using “ration test”;


# Main files to check
- Project report: I have briefly introduced the objectives of the project, reviewed the image processing methods, explained the main functions, described experiments and discussed the results.

- Notebook: High level code where inputs are given, main functions are called, results are displayed and saved.

- Student code: Image processing functions are defined.


# Setup by Dr. Kin-Choong Yow
- Install <a href="https://conda.io/miniconda.html">Miniconda</a>. It doesn't matter whether you use 2.7 or 3.6 because we will create our own environment anyways.
- Create a conda environment, using the appropriate command. On Windows, open the installed "Conda prompt" to run this command. On MacOS and Linux, you can just use a terminal window to run the command. Modify the command based on your OS ('linux', 'mac', or 'win'): `conda env create -f environment_<OS>.yml`
- This should create an environment named `ense885ay`. Activate it using the following Windows command: `activate ense885ay` or the following MacOS / Linux command: `source activate ense885ay`.
- Run the notebook using: `jupyter notebook ./code/proj2.ipynb`
- Generate the submission once you're finished using `python zip_submission.py`


# Credits and References
This project has been developed based on the project template and high-level code provided by Dr. Kin-Choong Yow, my instructor for the course “ENSE 885AY: Application of Deep Learning in Computer Vision”.

This course is based on Georgia Tech’s CS 6476 Computer Vision course instructed by James Hays.

- Dr. Kin-Choong Yow page: 
http://uregina.ca/~kyy349/

- “CS 6476 Computer Vision” page:
https://www.cc.gatech.edu/~hays/compvision/


- James Hays pages:
https://www.cc.gatech.edu/~hays/
https://github.com/James-Hays?tab=repositories


# My contribution:
The following files contain the code written by me:

- code/student_code.py >> get_interest_points() function >> Implantation of Harris Detector and ANMS 
- code/student_code.py >> get_features() function >> Implantation of SIFT-like algorithm
- code/student_code.py >> match_features() function >> Implantation of Feature Matching algorithm
- code/proj2.ipynb >> Effect of Scale Values on Correspondence Evaluation

______________
Marzieh Zamani