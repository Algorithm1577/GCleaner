import torch
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, global_mean_pool
import torch.nn.functional as F


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=32):
        super(GraphSAGE, self).__init__()

        dim = num_hidden

        self.conv1 = SAGEConv(num_features, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.conv2 = SAGEConv(dim, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.conv3 = SAGEConv(dim, dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.conv4 = SAGEConv(dim, dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.conv5 = SAGEConv(dim, dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.conv6 = SAGEConv(dim, dim)
        self.bn6 = torch.nn.BatchNorm1d(dim)

        self.conv7 = SAGEConv(dim, dim)
        self.bn7 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

        # explainability
        self.final_conv_acts = None
        self.final_conv_grads = None

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = F.relu(self.conv6(x, edge_index))
        x = self.bn6(x)
        with torch.enable_grad():
            self.final_conv_acts = self.conv7(x, edge_index)
        self.final_conv_acts.register_hook(self.activations_hook)
        x = F.relu(self.final_conv_acts)
        x = self.bn7(x)
        x = global_mean_pool(x, batch)
        # x = TopKPooling(x, batch)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
