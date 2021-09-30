import os
import pandas as pd

from lib.kmeans import KMeansModel
from lib.utils import standard_scale

# Path to the raw data
csv = "../../data/LKB_P.csv"
# Base path to results directory
results = "../../results/initial"

# Columns in the data set to use for clustering
process = ['E(HOMO)', 'E(LUMO)', 'He8_steric', 'PA', 'Q(B)', 'BE(B)', 'P-B', 'DP-A(B)', 'DA-P-A(B)', 'Q(Au)', 'BE(Au)',
           'Au-Cl', 'Au-P', 'DP-A(Au)', 'DA-P-A(Au)', 'Q(Pd)', 'BE(Pd)', 'Pd-Cl trans', 'P-Pd', 'DP-A(Pd)',
           'DA-P-A(Pd)', 'Q(Pt)', 'BE(Pt)', 'P-Pt', 'DP-A(Pt)', 'DA-P-A(Pt)', '<(H3P)Pt(PH3)', "S4'"]
# Columns in the data set to exclude from clustering
drop = ['Type', "PC1", "PC2", "PC3", "PC4"]

# IDs of reference ligands
pos_refs = [16, 41, 54, 113]
neg_refs = [21]

# Read the raw data
lkb = pd.read_csv(csv, sep=";", index_col=0)

# Data processing
X = lkb.drop(columns=drop)
X = standard_scale(X)

# Initialize the model
model = KMeansModel(X=X, k=8, rs=1)

# Run the model
model.run()

# Append the resulting clusters to the raw data and save it
lkb = pd.concat([lkb, model.clusters], axis=1)
lkb.to_csv(os.path.join(results, "LKB_P_clusters.csv"), sep=";")

# Run the statistical evaluation
stats = model.stats(ref_ids=pos_refs, k=8)
stats.to_csv(os.path.join(results, "LKB_P_stats.csv"), sep=";")

print(stats.loc[pos_refs])