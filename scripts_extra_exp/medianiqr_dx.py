"""
Script that computes information about the median and iqr of the number of timepoints
separated by DX
"""

import os
import sys
sys.path.insert(0, os.path.abspath('./'))
import numpy as np
import pandas as pd

seed = 1714
np.random.seed(seed)

# Data paths
path = "data/multimodal_no_petfluid.csv"

# Load data
df = pd.read_csv(path)

dx_dict = {
    "NL": "CN",
    "MCI": "MCI",
    "MCI to NL": "CN",
    "Dementia": "AD",
    "Dementia to MCI": "MCI",
    "NL to MCI": "MCI",
    "NL to Dementia": "AD",
    "MCI to Dementia": "AD"
}

df['DX'] = df['DX'].map(dx_dict)

df = df.sort_values(by=['PTID', 'Years_bl'], ascending=[True, True])

##Compute number of time points for different diagnosis
dict_of_dx = {"CN": [],
              "MCI": [],    
              "AD": []}

train_timesep = []
print('Train set')
print(len(np.unique(df["PTID"])))
for subject in np.unique(df["PTID"]):
    subject_df = df.loc[df["PTID"] == subject]

    #Get DX of the subject
    dx = subject_df["DX"].iloc[0]

    dict_of_dx[dx].append(len(subject_df))

#Compute median and iqr of each diagnosis
for dx in dict_of_dx.keys():
    print(dx)
    print("Median: {}".format(np.median(dict_of_dx[dx])))
    print("IQR: {}".format(np.percentile(dict_of_dx[dx], 75) -
                           np.percentile(dict_of_dx[dx], 25)))