# Cython
from lhrr import LhrrWrapper

# PIP packages
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MaxAbsScaler
from scipy.optimize import linear_sum_assignment as linear_assignment

# Torch packages
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# networks
from sgcc.networks import load_network, check_network

class Sgcc():

    def __init__(self,
                 k:int,
                 t:int,
                 p = 0.5,
                 network:str = "gcn",
                 neurons:int = 96,
                 metric: str = "euclidean",
                 verbose:bool = True):

        # Confirm network availability
        check_network(network)
        
        # Load parameters
        self.k = k
        self.t = t
        self.p = p
        self.edge_index = None
        self.anchor_ids = None
        self.scores = None
        self.pairs = None
        self.leaders = None
        self.network = network
        self.neurons = neurons
        self.metric = metric
        self.verbose = verbose

    def run(self,features: np.array, c:int) -> np.array:
        self.process_features(features)

        self.check_print("Finding leaders...")
        self.n_clusters = c
        self.find_leaders(self.n_clusters)
        self.clusters = []
        self.labels = np.full(self.n, -1)

        self.check_print("Initializing clusters...")
        for i in self.leaders:
            self.clusters.append(np.asarray([i], dtype=int))
            self.labels[i] = len(self.clusters) - 1
        
        self.check_print("Creating soft-labels...")
        targets = self.scores[:self.limit, 0].astype(int)
        for i in targets:
            if i not in self.leaders:
                self.classify(i)

        self.check_print("Running GCN...")
        self.initial_labels = self.labels
        self.initial_clusters = self.clusters
        return self.run_gcn()

    def process_features(self, features:np.array) -> None:
        self.features = features
        self.n = len(features)
        self.limit = int(self.n * self.p)
        self.run_lhrr()
    
    def run_lhrr(self):
        self.check_print("Running LHRR...")
        self.lhrr = LhrrWrapper(self.k, self.t)
        self.lhrr.run(self.features, metric=self.metric)
        self.hyperedges = self.lhrr.get_hyper_edges()
        self.confid = self.lhrr.get_confid_scores()
        self.ranked_lists = self.lhrr.get_ranked_lists()
    
    def check_print(self, msg:str) -> None:
        if self.verbose:
            print(msg)
    
    def find_leaders(self, k=int) -> None:
        if self.scores is None:
            self.compute_scores()

        leaders = []
        self.compute_he_matrix()
        self.leader_intersection_score = [np.zeros(self.n) for i in range(k)]
        leaders.append(int(self.scores[0, 0]))
        for i in range(1, k):
            leaders.append(self.find_next_leader(leaders))
        self.leaders = leaders
    
    def compute_scores(self):
        scores = []
        for i in range(self.n):
            score = self.confid[i] * self.get_self_score(i)
            scores.append([i, score])
        scores.sort(key=lambda x: x[1])
        scores.reverse()
        self.scores = np.asarray(scores)
    
    def get_self_score(self, i:int):
        he = self.hyperedges[i]
        for pair in he:
            if int(pair[0]) == i:
                return pair[1]
        return None
    
    def compute_he_matrix(self):
        he_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            he = self.hyperedges[i]
            for item, score in he:
                he_matrix[i][item] = score
        self.he_matrix = he_matrix
    
    def find_next_leader(self, leaders):
        compute_target = len(leaders) - 1
        scores = []
        for idx, score in self.scores:
            idx = int(idx)
            if idx in leaders:
                continue
            div = 0
            for pos, i in enumerate(leaders):
                div += self.compute_intersection_score(i, idx, pos, pos == compute_target)
            scores.append([idx, score/(1+div)])
        scores.sort(key=lambda x: x[1])
        scores.reverse()
        return int(scores[0][0])
    
    def compute_intersection_score(self, leader:int, target:int, leader_pos:int, compute:bool):
        if compute:
            total_score = np.dot(self.he_matrix[target], self.he_matrix[leader])
            self.leader_intersection_score[leader_pos][target] = total_score
            return total_score
        else:
            return self.leader_intersection_score[leader_pos][target]
        
    def classify(self, item:int):
        scores = []
        for leader in self.leaders:
            scores.append(self.classification_score(item, leader))
        scores.sort(key=lambda x: x[1])
        scores.reverse()
        
        leader = scores[0][0]
        target = self.labels[leader]
        
        self.clusters[target] = np.append(self.clusters[target], item)
        self.labels[item] = target
    
    def classification_score(self, item:int, leader:int):
        cluster = self.clusters[self.labels[leader]]
        s = 0
        for i in cluster:
            s += self.he_matrix[i][item]
        return [leader, (s/len(cluster))]
    
    def run_gcn(self) -> np.array:
        
        # Load initial configuration
        self.labels = self.initial_labels
        self.clusters = self.initial_clusters

        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data definition
        y = torch.tensor(self.labels).to(device)
        x = torch.tensor(self.features).to(device)

        # Mask definition
        train_mask = torch.tensor(self.labels != -1).to(device)
        val_mask = torch.tensor([]).to(device)
        test_mask = torch.tensor([]).to(device)
        
        # Edge index
        edge_index = self.compute_edge_index(self.k)
        edge_index = torch.tensor(edge_index)
        edge_index = edge_index.contiguous().to(device)
        self.edge_index = edge_index

        # Tensor data
        data = Data(x=x.float(),
                    edge_index=edge_index,
                    y=y,
                    test_mask=test_mask,
                    train_mask=train_mask,
                    val_mask=val_mask)

        # Variables
        pNFeatures = len(self.features[0])
        pNEpochs = 400

        
        pLR = 0.001
        model = load_network(pNFeatures,
                             self.neurons,
                             self.n_clusters,
                             self.network).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=pLR,
                                     weight_decay=5e-4)

        model.train()
        for epoch in range(pNEpochs):
            if epoch+1 % 100 == 0:
                # Divide pLR by 2 every 100 epochs
                pLR /= 2
            optimizer.zero_grad()
            out = model(data)

            # Overfit checking
            _, pred = out.max(dim=1)
            correct = float(pred[data.train_mask]
                        .eq(data.y[data.train_mask])
                        .sum()
                        .item())
            acc = correct / data.train_mask.sum().item()
            if acc == 1.0:
                self.check_print(f"Early stoping on epoch {epoch}")
                break

            loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        _, pred = model(data).max(dim=1)
            
        # Training acc
        correct = float(pred[data.train_mask]
                        .eq(data.y[data.train_mask])
                        .sum()
                        .item())
        acc = correct / data.train_mask.sum().item()
        self.check_print(f"- training acc: {acc}")

        pred = pred.cpu()
        self.labels_ = pred.numpy()
        return self.labels_
    
    def run_gcn_nada(self,
                     iterations:int=10):
        
        # Load initial configuration
        self.labels = self.initial_labels
        self.clusters = self.initial_clusters

        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data definition
        y = torch.tensor(self.labels).to(device)
        x = torch.tensor(self.features).to(device)

        # Mask definition
        train_mask = torch.tensor(self.labels != -1).to(device)
        val_mask = torch.tensor([]).to(device)
        test_mask = torch.tensor([]).to(device)
        
        # Edge list
        if self.edge_index is None:
            edge_index = self.compute_edge_index(self.k)
            self.edge_index = edge_index
        else:
            edge_index = self.edge_index
        
        if type(edge_index) != torch.Tensor:
            edge_index = torch.tensor(edge_index)
        
        edge_index = edge_index.contiguous().to(device)

        # Tensor data
        data = Data(x=x.float(),
                    edge_index=edge_index,
                    y=y,
                    test_mask=test_mask,
                    train_mask=train_mask,
                    val_mask=val_mask)

        # Variables
        pNFeatures = len(self.features[0])
        pNEpochs = 400

        
        # Training + evaluation
        train_accs = []
        nmis = []
        rands = []
        vs = []
        accs = []
        for i in range(iterations):
            # Initialize learning rate in 1e-3
            pLR = 0.001
            model = load_network(pNFeatures,
                                 self.neurons,
                                 self.n_clusters,
                                 self.network).to(device)
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=pLR,
                                         weight_decay=5e-4)

            model.train()
            for epoch in range(pNEpochs):
                if epoch+1 % 100 == 0:
                    # Divide pLR by 2 every 100 epochs
                    pLR /= 2
                optimizer.zero_grad()
                out = model(data)

                # Overfit checking
                _, pred = out.max(dim=1)
                correct = float(pred[data.train_mask]
                            .eq(data.y[data.train_mask])
                            .sum()
                            .item())
                acc = correct / data.train_mask.sum().item()
                if acc == 1.0:
                    print(f"Early stoping on epoch {epoch}")
                    print("- acc:", acc)
                    break

                loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask])
                loss.backward()
                optimizer.step()

            model.eval()
            _, pred = model(data).max(dim=1)
            
            # Training acc
            correct = float(pred[data.train_mask]
                            .eq(data.y[data.train_mask])
                            .sum()
                            .item())
            acc = correct / data.train_mask.sum().item()
            train_accs.append(acc)
            print("- acc:", acc)

            pred = pred.cpu()
            self.labels = pred.numpy()
            nmis.append(
                metrics.normalized_mutual_info_score(
                    self.classes,
                    pred,
                    average_method='min'))
            vs.append(metrics.v_measure_score(self.classes, pred))
            accs.append(self.cluster_acc(self.classes, pred.numpy()))
        
        nmis = np.asarray(nmis)
        vs = np.asarray(vs)
        accs = np.asarray(accs)

        nmi_mean = np.mean(nmis)
        nmi_std = np.std(nmis)
        vs_mean = np.mean(vs)
        vs_std = np.std(vs)
        acc_mean = np.mean(accs)
        acc_std = np.std(accs)

        print("Results:")
        print("Training acc ->", np.mean(train_accs), "+-", np.std(train_accs))
        print("NMI score -> ", nmi_mean, "+-", nmi_std)
        print("V-Measure -> ", vs_mean, "+-", vs_std)
        print("ACC -> ", acc_mean, "+-", acc_std)

        return nmi_mean, nmi_std, vs_mean, vs_std, acc_mean, acc_std
    
    def compute_edge_index(self, k:int=25):
        edge_list = []
        for i in range(self.n):
            rank = self.ranked_lists[i]
            # Loop over k+1 positions, since the item itself is one of them
            for j in range(k+1):
                if (rank[j] != i) and self.is_reciprocal(rank[j], i, k):
                    edge_list.append((i, rank[j]))
        return np.asarray(edge_list).transpose()

    def is_reciprocal(self, anchor:int, target:int, k:int):
        for i in range(k):
            if self.ranked_lists[anchor][i] == target:
                return True
        return False

    def evaluate(self, y_true, y_pred):
        nmi = metrics.normalized_mutual_info_score(
                y_true,
                y_pred,
                average_method='min')
        vscore = metrics.v_measure_score(y_true, y_pred)
        acc = self.cluster_acc(y_true, y_pred)

        return nmi, vscore, acc

    @staticmethod
    def cluster_acc(y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        acc = np.sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size
        return acc

    



