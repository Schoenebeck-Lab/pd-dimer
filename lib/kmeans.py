import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist


class KMeansModel:

    def __init__(self, X, k=8, rs=1):
        """
        Wrapper for the sklearn.KMeans algorithm that extracts relevant information.. Provides functionality to optimize
        and build the model.

        :param X:   (pd.DataFrame) The (processed) input data for the model. Needs to be provided as a pandas.DataFrame.
                                   Ideally, the index of the DataFrame refers to the identifiers within the data set.
        :param k:   (int) Cluster number to be used for the kmeans algorithm. Default: 8.
        :param rs:  (int) Random state to be used for the model. Default: 1
        """
        self.X = X
        self.k = k
        self.rs = rs

        # Storage for model
        self.model = None

        # Storage for results
        self.clusters = pd.DataFrame()

        # Storage for evaluation metrics
        self.inert = float  # Inertia
        self.dist = float  # Distortion
        self.sil = float  # Average Silhouette Score
        self.sil_sample = None  # Silhouette Score per sample

    def run(self):
        """
        Run the kmeans algorithm on the given data set with a given cluster number k and a given random state.
        """
        self.model = KMeans(n_clusters=self.k, n_init=10, init="k-means++", random_state=self.rs)
        self.model.fit(self.X)

        self.clusters = pd.DataFrame({"Cluster": self.model.labels_}, index=self.X.index)
        self.inert = float(self.model.inertia_)
        self.dist = float(sum(np.min(cdist(self.X.to_numpy(), self.model.cluster_centers_, 'euclidean'), axis=1)) /
                          self.X.shape[0])
        self.sil = float(silhouette_score(self.X.to_numpy(), self.model.labels_))
        self.sil_sample = silhouette_samples(self.X.to_numpy(), self.model.labels_)

    def opt(self, ks):
        """
        Find the optimal cluster number k from a list of different ks.

        :param ks:  (list) Values for k that should be checked during the optimization.
        :return:    metrics (pd.DataFrame): Inertia, Distortion and Average Silhouette scores for each value of k
                    clusters (pd.DataFrame): The resulting clusters for each value of k.
                    sil_samples (pd.DataFrame): Per sample silhouette score for each value of k.
        """
        metrics = pd.DataFrame(columns=["Inertia", "Distortion", "Silhouette"])
        metrics.index.name = "k"

        clusters = pd.DataFrame(index=self.X.index)
        clusters.index.name = self.X.index.name
        sil_samples = pd.DataFrame(index=self.X.index)
        sil_samples.index.name = self.X.index.name

        for k in tqdm(ks):
            self.k = k
            self.run()
            metrics.at[k, "Inertia"] = self.inert
            metrics.at[k, "Distortion"] = self.dist
            metrics.at[k, "Silhouette"] = self.sil

            clusters[f"k={k}"] = self.clusters.to_numpy().flatten()
            sil_samples[f"k={k}"] = self.sil_sample

        return metrics, clusters, sil_samples

    def stats(self, ref_ids, k=None, rs_range=None):
        """
        Investigate the dependency of the resulting clusters on different random_states. The function runs the k-means
        algorithm for each provided random state and checks how often each instance is grouped with the provided
        reference instances and calculates a normalized score.

        :param ref_ids:     (list) List of indices of the references, for which the similarity to other instances in the
                            data should be evaluated. The indices must correspond to the index in self.X!
        :param k:           (int) Cluster number k. Default: None (= use the value stored in self.k)
        :param rs_range:    (list) List of random states that should be checked. Default: None (= use range(0, 1000))

        :return:    stats (pd.DataFrame)    Table containing information on whether an instance was grouped with the
                                            references or not when using the provided random states (One-Hot-Encoded).
                                            The last column ("Score") contains the percentage of times the instance was
                                            grouped with the references.
        """
        # Range of random states to check
        rs_range = range(0, 1000) if rs_range is None else rs_range

        # Set the value for k, if provided
        self.k = k if k is not None else self.k

        # Storage
        stats = pd.DataFrame(index=self.X.index)

        # For each random state ...
        for rs in tqdm(rs_range):
            # Define a column name
            col = f"RS{rs}"
            # Run the model
            self.rs = rs
            self.run()
            # Report the resulting clusters
            stats[col] = self.clusters.to_numpy().flatten()
            # Get the clusters that contain references
            ref_cluster = list(set(stats.loc[stats.index.isin(ref_ids)][col]))
            # Get all members within these clusters
            ref_cluster_members = list(stats.loc[stats[col].isin(ref_cluster)].index)
            # One Hot Encode the results (1: instance was clustered with refs, 0: instance was not clustered with refs)
            for idx in stats.index:
                stats.at[idx, col] = 1 if idx in ref_cluster_members else 0

        # Get the total number of times a ligand was clustered with the references
        stats["Sum"] = stats.sum(axis=1)
        # Calculate percentage (= Score)
        stats["Score"] = stats["Sum"] / len(rs_range)

        return stats



