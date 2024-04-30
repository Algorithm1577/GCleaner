from typing import List
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree


def trian(model, optimizer, criterion, train_loader, device):
    model.train()
    loss_all = 0
    graph_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        model.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        y = data.y
        loss = criterion(output, y).to(device)
        # loss = (loss - args.gamma)**2

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        graph_all += data.num_graphs
        optimizer.step()
    loss = loss_all / graph_all

    return loss


def test(model, criterion, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    max_loss = -np.inf
    min_loss = np.inf
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]

        y = data.y
        loss += criterion(output, y).item() * data.num_graphs
        loss_batch = criterion(output, y).item()
        if loss_batch > max_loss:
            max_loss = loss_batch
        if loss_batch < min_loss:
            min_loss = loss_batch
        correct += pred.eq(y).sum().item()
        total += data.num_graphs
    acc = correct / total
    loss = loss / total

    return acc, (loss, max_loss, min_loss)


def stat_graph(graphs_list: List[Data]):
    num_total_nodes = []
    num_total_edges = []
    for graph in graphs_list:
        num_total_nodes.append(graph.num_nodes)
        num_total_edges.append(graph.edge_index.shape[1])
    avg_num_nodes = sum(num_total_nodes) / len(graphs_list)
    avg_num_edges = sum(num_total_edges) / len(graphs_list) / 2.0
    avg_density = avg_num_edges / (avg_num_nodes * avg_num_nodes)

    median_num_nodes = np.median(num_total_nodes)
    median_num_edges = np.median(num_total_edges)
    median_density = median_num_edges / (median_num_nodes * median_num_nodes)

    max_num_nodes = max(num_total_nodes)

    return (
        avg_num_nodes,
        avg_num_edges,
        avg_density,
        median_num_nodes,
        median_num_edges,
        median_density,
        max_num_nodes,
    )


def grad_cam(final_conv_acts, final_conv_grads):
    node_heat_map = []
    alphas = torch.mean(
        final_conv_grads, axis=0
    )  # mean gradient for each feature (512x1)
    for n in range(final_conv_acts.shape[0]):  # nth node
        node_heat = F.relu(alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
    return node_heat_map


def load_data(name):
    if name == "AIDS" or name == "MUTAG":
        dataset = TUDataset(root="./data", name=name)
        num_classes = dataset.num_classes
        dataset = list(dataset)
    else:
        raise ValueError("Unknown dataset")
    print("# data len:", len(dataset))
    print(f"Number of classes: {num_classes}")
    return dataset, num_classes


def degree_centrality(data):
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)

    deg_centrality = deg / (data.num_nodes - 1)
    centrality_dict = {
        i: deg_c.item() for i, deg_c in zip(range(data.num_nodes), deg_centrality)
    }

    return centrality_dict
