"""
Script that computes information about the median and iqr of the time
separation between the acquisitions of the same subject in the dataset, both
train and test
"""

import os
import sys
sys.path.insert(0, os.path.abspath('./'))
import numpy as np
import pandas as pd

seed = 1714
np.random.seed(seed)

# Load data
# Data paths
train_path = "data/multimodal_no_petfluid_train.csv"
test_path = "data/multimodal_no_petfluid_test.csv"

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Get the time separation between acquisitions of the same subject
# Time intervals between consecutive acquisitions of the same subject
# in the train set
train_timesep = []
print('Train set')
print(len(np.unique(train_df["PTID"])))
for subject in np.unique(train_df["PTID"]):
    subject_df = train_df.loc[train_df["PTID"] == subject]
    subject_df = subject_df.sort_values(by="Years_bl")
    for i in range(subject_df.shape[0] - 1):
        train_timesep.append(subject_df.iloc[i + 1]["Years_bl"] -
                             subject_df.iloc[i]["Years_bl"])

# Time intervals between consecutive acquisitions of the same subject
# in the test set
test_timesep = []
print('Train set')
print(len(np.unique(test_df["PTID"])))
for subject in np.unique(test_df["PTID"]):
    subject_df = test_df.loc[test_df["PTID"] == subject]
    subject_df = subject_df.sort_values(by="Years_bl")
    for i in range(subject_df.shape[0] - 1):
        test_timesep.append(subject_df.iloc[i + 1]["Years_bl"] -
                            subject_df.iloc[i]["Years_bl"])

# train_timesep and test_timesep are in years. Convert it to months
train_timesep = np.array(train_timesep) * 12
test_timesep = np.array(test_timesep) * 12

# Compute median and iqr first of the train dataset, and then of the test set
# and print the results
print("Train set")
print("Median: {}".format(np.median(train_timesep)))
print("IQR: {}".format(np.percentile(train_timesep, 75) -
                       np.percentile(train_timesep, 25)))

print("Test set")
print("Median: {}".format(np.median(test_timesep)))
print("IQR: {}".format(np.percentile(test_timesep, 75) -
                          np.percentile(test_timesep, 25))) 


# Compute the median and iqr of the time separation, combining test and train
# and print the results
print("Train + test set")
timesep = np.concatenate([train_timesep, test_timesep])
median = np.median(timesep)
iqr = np.percentile(timesep, 75) - np.percentile(timesep, 25)

# Print the results
print("Median time separation: {}".format(median))
print("IQR time separation: {}".format(iqr))

