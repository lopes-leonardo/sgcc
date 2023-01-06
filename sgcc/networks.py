import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SGConv, APPNP

class GCNNet(torch.nn.Module):
    def __init__(self, features, neurons, classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(features, neurons) #dataset.num_node_features
        self.conv2 = GCNConv(neurons, classes) #dataset.num_classes
        self.embedding = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        self.embedding = x
        return F.log_softmax(x, dim=1)

class SGCNet(torch.nn.Module):
    def __init__(self, features, neurons, classes):
        super(SGCNet, self).__init__()
        self.conv1 = SGConv(features, neurons, K=2) #dataset.num_node_features
        self.conv2 = SGConv(neurons, classes, K=2) #dataset.num_classes
        self.embedding = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        self.embedding = x
        return F.log_softmax(x, dim=1)


class SGCNet2(torch.nn.Module):
    def __init__(self, features, neurons, classes):
        super(SGCNet2, self).__init__()
        self.conv1 = SGConv(features, classes, K=2)
        self.embedding = None

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        self.embedding = x
        return F.log_softmax(x, dim=1)

class APPNPNet(torch.nn.Module):
    def __init__(self, features, neurons, classes):
        super(APPNPNet, self).__init__()
        self.lin1 = Linear(features, neurons)
        self.lin2 = Linear(neurons, classes)
        self.prop1 = APPNP(10, 0.1)
        self.embedding = None

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        self.embedding = x
        return F.log_softmax(x, dim=1)


networks = {
    'gcn': GCNNet,
    'sgc': SGCNet2,
    'appnp': APPNPNet,
}

def check_network(network:str):
    if network not in networks.keys():
        raise Exception(f'Network {network} not found. Available options: \
{networks.keys()}')

def load_network(features:int, neurons:int, classes:int, network:str):
    check_network(network)
    return networks[network](features, neurons, classes)