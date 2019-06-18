# soundSourcePresenceEstimation

This repository contains the replication material for *Estimation of the perceived time of presence of sources in urban acoustic environments using deep learning techniques*.

The *python* folder contains the deep learning model implementation, and the *matlab* folder contains data processing as well as all other experiments in the paper.

# Getting started

1. Clone or download repository from [Github](https://github.com/felixgontier/soundSourcePresenceEstimation).
2. Download the experiment corpus from [Zenodo](https://zenodo.org/record/3248734#.XQjC4v7gqUk) and extract its contents to *matlab/*.
3. Download the deep learning dataset from [Zenodo](https://zenodo.org/record/3248703#.XQjDVv7gqUk) and extract its contents to *python/data/*.
4. Install requirements: ''pip3 install -r requirements.txt''

# Usage

1. Run ''presProfileDeep.m'' to generate ground truth presence labels.
2. Copy audio files from *matlab/audio/rec* and *matlab/audio/rep* to *python/data/test_recrep/sound*, and from *matlab/audio/sim* to *python/data/test_sim/sound*.
3. Run ''python3 main.py'' to train the model, compute performance metrics on the evaluation dataset, and generate presence predictions for the perceptual experiment corpus.
4. Copy *test_recrep_pred.txt* and *test_sim_pred.txt* from *python* to *matlab*.
5. Run ''paperReplication.m'' to replicate the paper results.

## Contact
- **Felix Gontier** (<felix.gontier@ls2n.fr>)
- **Mathieu Lagrange** (<mathieu.lagrange@cnrs.fr>)
