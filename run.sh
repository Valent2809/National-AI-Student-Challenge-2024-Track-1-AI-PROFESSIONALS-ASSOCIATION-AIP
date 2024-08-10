#!/bin/bash

# Run dataprep.py to preprocess the data
python src/dataprep/dataprep.py

# Run model.py to train the model and save it
python src/model/model.py
