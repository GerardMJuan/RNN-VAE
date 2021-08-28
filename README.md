## MC-RVAE README

This repository contains code for the project of multi-channel recurrent variational neural networks for Alzheimer's Disease progression.

If you are reading this, beware of uncommented code and a general disorder on the scripts. My apologies. I suffer from chronic code disorder
and [this is not a joke](https://youtu.be/RC5imk9sG1M?t=4). I am available by email for any issue you may find on the code.

Table of contents:
* evaluation/: scripts to evaluate the results. Contains some Jupyter notebooks to generate figures, and Python scripts to test the results.
* rnnvae/: main scripts of the method. Contains the definition of the model and supporting functions. It has 3 different definitions of the model:
  * rnnvae: Full model.
  * rnnvae_h: Model with sharing the RNN and prior part.
  * rnnvae_s: Model without phi_x and phi_z functions.  <-- CURRENT, MOST ADVANCED MODEL. IGNORE THE COMMENTS

Scripts folder:

* scripts_mc/: Scripts for testing with synthetic and real brain data, using rnnvae.
* scripts_mc_moreparams/: Scripts for testing with synthetic and real brain data, using rnnvae_h and with separated parameters.
* scripts_small/: Scripts for testing with synthetic and real brain data, using rnnvae_s.
