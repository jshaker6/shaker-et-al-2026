# Analysis code for Shaker, Schroeter, Birman, & Steinmetz (2026)

Code accompanying the paper: **The midbrain reticular formation in contextual control of perceptual decisions**.

## Data

Associated data files are available on FigShare:  
[https://doi.org/10.6084/m9.figshare.31428575](https://doi.org/10.6084/m9.figshare.31428575)

Behavioral modeling code is in a separate repository:  
[https://github.com/jeremyschroeter/mrf-behavior](https://github.com/jeremyschroeter/mrf-behavior)

## Repository contents

### Demo scripts

Two self-contained demo scripts walk through the main analyses on an example session:

- **`demo_GLM_encoding.m`** - Kernel regression encoding model. Fits time-lagged kernels of task variables to each neuron's firing rate, then assesses variable significance via shuffling-based method.
- **`demo_decoding_trajectories.m`** - Context decoding accuracy and population trajectory analyses. Decodes the latent context variable from pre-stimulus baseline activity, fits Action/Choice/Context coding dimensions, and projects population activity onto these dimensions to visualize neural trajectories.

### Helper functions

- **`loadSessionData.m`** - Loads all data for one recording session from the FigShare folder structure (behavioral data, spike sorting outputs, etc).
- Additional helper functions for spike preprocessing, predictor construction, cross-validation, etc.

## Dependencies

- Tested on MATLAB version R2020a
- [glmnet_matlab](https://github.com/lachioma/glmnet_matlab) - Fast elastic net / ridge regression
    - Inlcuded in helpers folder
- [npy-matlab](https://github.com/kwikteam/npy-matlab) - Reading `.npy` files
    - Included in helpers folder
- MATLAB Statistics and Machine Learning Toolbox
- MATLAB Signal Processing Toolbox
- MATLAB Parallel Computing Toolbox
