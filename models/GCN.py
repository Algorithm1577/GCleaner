import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_add_pool
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=32):
        super(GCN, self).__init__()

        dim = num_hidden

        self.conv1 = GCNConv(num_features, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.conv2 = GCNConv(dim, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.conv3 = GCNConv(dim, dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.conv4 = GCNConv(dim, dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.conv5 = GCNConv(dim, dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

        # explainability
        self.final_conv_acts = None
        self.final_conv_grads = None

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        x = self.conv1(x, edge_index, edge_weights)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_weights)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        x = F.relu(x)

        with torch.enable_grad():
            self.final_conv_acts = self.conv3(x, edge_index, edge_weights)
        self.final_conv_acts.register_hook(self.activations_hook)
        x = torch.nn.functional.normalize(self.final_conv_acts, p=2, dim=1)
        x = F.relu(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
