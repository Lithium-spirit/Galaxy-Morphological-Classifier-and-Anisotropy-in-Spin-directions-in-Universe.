# Galaxy-Morphological-Classifier-and-Anisotropy-in-Spin-directions-in-Universe.

This Project is a pipeline for classifying an image of a galaxy into spiral/elliptical/edge on and if spiral detects the spin direction.

I have used Shamir's ganalyzer algorithm mentioned in https://arxiv.org/pdf/1105.3214 to get the properties of the galaxies, generate radial intensity plots and then detect peaks.
I have grouped these peaks to highlight the arm structure of the galaxy.
I use the properties of this arm structure such as slope and standard deviation to classify the galaxy and the spin direction of the galaxy.

Using the data obtained from the algorithm above I use Bayesian analysis to test the anisotropy in spin direction of spiral galaxies in the universe thus questioning the very foundation of modern cosmology.
I test the the dependence of quadrupole, dipole and monopole models. The Anisotropy.py file in my repository is currently using the available dataset by Iye. However the data generated from gan.ipynb can be inputted for analysis by changing a few variables.

I have used emcee to perform Bayesian Interference on the dataset with varying models and used BIC as the criteria for benchmark.
