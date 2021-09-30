import os
import pandas as pd

from lib.pca import PCAModel
from lib.utils import min_max_scale

# Path to the raw data
csv = "../../data/Dimer_LKB_P.csv"
# Base path to results directory
results = "../../results/dimer-specific/pca"

# Columns in the data set to use for PCA
process = ['E(HOMO)', 'E(LUMO)', 'He8_steric', 'PA', 'Q(Pd)', 'BE(Pd)',
           'Pd-Cl trans', 'P-Pd', 'DP-A(Pd)', 'DA-P-A(Pd)', 'BDE1', 'BDE2', 'BDE3',
           'COMP1', 'DIM1', 'REORG1', 'REDOX1', 'REDOX2', 'REDOX3', 'REDOX4',
           'A_NBO_P', 'B_NBO_Pd', 'B_NBO_P', 'B_LENGTH_Pd_P', 'B_B1', 'B_lval',
           'B_newB5', 'C_NBO_Pd', 'C_NBO_P', 'C_LENGTH_Pd_P', 'C_ANGLE_P_Pd_P',
           'D_NBO_Pd', 'D_NBO_P', 'D_LENGTH_Pd_P', 'D_LENGTH_Pd_Pd',
           'D_ANGLE_P_Pd_P', 'D_ORDER_Pd_Pd', 'E_NBO_Pd', 'E_NBO_P',
           'E_LENGTH_Pd_P', 'E_ANGLE_I_Pd_I', 'F_NBO_Pd', 'F_NBO_P',
           'F_LENGTH_Pd_P', 'F_ANGLE_P_Pd_P', 'F_ANGLE_I_Pd_I', 'G_NBO_Pd',
           'G_NBO_P', 'G_LENGTH_Pd_P', 'G_LENGTH_Pd_Pd', 'G_TORSION_Pd_I_I_Pd',
           'G_ORDER_Pd_Pd']

# Columns in the data set to exclude from PCA
drop = ["Type", "PC1", "PC2", "PC3", "PC4", "PC1'", "PC2'", "PC3'", "PC4'"]

# IDs of reference ligands
pos_refs = [16, 41, 54, 113]
neg_refs = [21]

# Read the raw data
dimer = pd.read_csv(csv, sep=";", index_col=0)

# Data processing
X = dimer.drop(columns=drop)
X = min_max_scale(X)

# Run PCA
model = PCAModel(X=X, n=4, rs=1)

# Save the results
model.pcs.to_csv(os.path.join(results, "pcs.csv"))
model.load.to_csv(os.path.join(results, "loadings.csv"))
model.summary.to_csv(os.path.join(results, "summary.csv"))


