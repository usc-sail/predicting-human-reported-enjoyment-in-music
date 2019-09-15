# Predicting-Human-Reported-Enjoyment-In-Music
Experiments written in Python to determine which models and auditory features best predict human-reported enjoyment in music.
 
# Background
This repository contains Python code used in the project "Predicting Human-Reported Enjoyment Responses in Happy and Sad Music" by Ma, B., Greer, T., Sachs, M., Kaplan, J., and Narayanan, S. The paper can be found in the proceedings of the 8th International Conference on Affective Computing and Intelligent Interaction, Cambridge, UK, September 2019.

# Usage
For experiments testing [Temporal Pattern Attention-LSTM (TPA-LSTM)](https://github.com/gantheory/TPA-LSTM), use file `main.py` in the TPA-LSTM folder. TPA-LSTM lib code has been modified from the original repository.

For experiments testing Autoregressive and Distributed Lag models, including Lasso, Ridge, and ARIMA, see the "AR-VAR" folder. Details on each model used can be found in their eponymous .py file; the overall testing suite can be run with `run_experiments.py`.

# Note on Data Used
To protect the privacy of the participants of the study, we did not include any of the data collected in this repository. If you wish to view it, contact me at benjamjm@usc.edu.
