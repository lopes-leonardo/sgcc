from sgcc.clustering import Sgcc
import numpy as np
import math

# Parameters
K = 85
T = 2
P = 0.5
NETWORK = "sgc"
NEURONS = 96
METRIC = "euclidean"
VERBOSE = True
C = 17

# Load features and classes
features = np.load(f"./data/flowers-resnet.npy", allow_pickle=True)
classes = np.asarray([math.floor(i/80) for i in range(len(features))])

# Run SGCC
cluster = Sgcc(K,
               T,
               p=P,
               network=NETWORK,
               neurons=NEURONS,
               metric=METRIC,
               verbose=VERBOSE)
cluster_labels = cluster.run(features, C)

# Evaluate results
nmi, vscore, acc = cluster.evaluate(classes, cluster_labels)
print("NMI ->", nmi)
print("V_Measure ->", vscore)
print("Accuracy ->", acc)