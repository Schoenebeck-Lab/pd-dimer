import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score

import matplotlib.pyplot as plt


class PCAModel:

    def __init__(self, X, n=4, rs=1):
        """
        Wrapper for the sklearn.PCA algorithm that extracts relevant information.

        :param X:   (pd.DataFrame) The (processed) input data for the model. Needs to be provided as a pandas.DataFrame.
                                   Ideally, the index of the DataFrame refers to the identifiers within the data set.
        :param n:   (int) Number of PCs. Default: 4
        :param rs:  (int) Random state to be used for the model. Default: 1
        """

        self.X = X
        self.n = n
        self.rs = rs

        # Storage for model
        self.model = None
        # Storage for resulting PCs
        self.pcs = pd.DataFrame()
        # Storage for PC Loadings
        self.load = pd.DataFrame()
        # Storage for Summary
        self.summary = pd.DataFrame()

        self.run()

    def run(self):
        """
        Run the principal component analysis with the provided parameters.
        """
        self.model = PCA(n_components=self.n, random_state=self.rs)

        names = [f"PC{i + 1}" for i in range(self.n)]

        self.pcs = pd.DataFrame(self.model.fit_transform(self.X), columns=names, index=self.X.index)

        self.load = pd.DataFrame(self.model.components_.T, columns=names, index=self.X.columns)

        self.summary = pd.DataFrame({
            "Variance": self.model.explained_variance_ratio_,
            "Cumulative Variance": self.model.explained_variance_ratio_.cumsum(),
            "Singular Value": self.model.singular_values_
            }, index=names)




