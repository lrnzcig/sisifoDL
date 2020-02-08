import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import sys

# TODO TODO TODO refactor from_distance_matrix.py
distance_matrix = pd.read_csv("data/dtw_distance_matrix_16K_upper_cluster_shorttype.csv")
print(sys.getsizeof(distance_matrix))

distance_matrix.sort_values(by="CUENTA1")
np.max(distance_matrix['CUENTA1'])
np.min(distance_matrix.columns)
distance_matrix[str(np.max(distance_matrix['CUENTA1']))] = 0
dict_row = {**{"CUENTA1": int(np.min(distance_matrix.columns))}, **{key: 0 for key in distance_matrix.columns}}
distance_matrix = distance_matrix.append(pd.DataFrame(dict_row,
                                                      index=[distance_matrix.shape[0]]), sort=True)
distance_matrix = distance_matrix.sort_values(by="CUENTA1").set_index("CUENTA1")
distance_matrix.index.name = None
print(distance_matrix)

# convertir en matriz de distancias simetrica
distance_matrix_symmetric = distance_matrix.to_numpy(np.int32) + distance_matrix.transpose().to_numpy(np.int32)
print(type(distance_matrix_symmetric[0, 0]))
print(pd.DataFrame(distance_matrix_symmetric))
print(sys.getsizeof(distance_matrix_symmetric))

# clusters
n_clusters = 8
acm = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete', affinity='precomputed')
clustering = acm.fit(distance_matrix_symmetric)
counts = Counter(clustering.labels_)
print(counts)