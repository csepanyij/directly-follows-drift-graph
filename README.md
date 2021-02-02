# Directly-Follows Drift Graph
This repository contains a modified Directly-Follows Graph designed to analyze process drifts.

## Installation
Run the following command to install all necessary libraries

<code>
cd path/to/folder
pip install -r requirements.txt
</code>

## Execution
There are two ways to run the package: via the Jupyter notebook or via script execution.

### Jupyter notebook
If you would like to examine your process log file in detail while playing around and experimenting with the log, then copy the ```process_drift_analysis.ipynb``` to your Jupyter environment and put the .xes files in the same directory.

### Script execution
Using the ```process_drift.py``` script, the complete code runs all at once, providing the DFDG the quickest way. However, the parameters are defined in the beginning of the script file, which need to be modified.

- ```INPUT_XES_FILE```: The input file to be read, with relative pathing to the working directory.
- ```OUTPUT_GRAPH_FILE```: Name of the output DFDG graph image (without extension).
- ```PENALTY```: Sensitivity parameter for Change-point detection, lower number means more sensitivity, more change points and more time preiods to choose from. Default is 4.
- ```MODEL```: The type of statistical model to be used for determining change points. Default is 'rbf'. More info: https://github.com/deepcharles/ruptures.
- ```CHOSEN_PERIOD1```: The first (earlier) time period for the comparison.
- ```CHOSEN_PERIOD2```: The second (later) time period for the comparison.
- ```ACTIVITY_NAMES```: Variable name in .xes input file, which contains the activity names. These will be plotted in the nodes of the output DFG.