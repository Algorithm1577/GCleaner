import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import degree
from utils import degree_centrality
import torch.nn as nn
import torch.nn.functional as F
import copy


def inject_sub_trigger(
    args,
    dataset,
    mode="ER",
    inject_ratio=0.1,
    backdoor_num=4,
    target_label=1,
    density=0.8,
):
    """
    Inject a sub trigger into the clean graph, return the poisoned dataset
    :param inject_ratio:
    :param dataset:
    :param mode:
    :return:
    """
    if mode == "ER":
        G_gen = nx.erdos_renyi_graph(backdoor_num, density, seed=args.seed)
    else:
        raise NotImplementedError

    print("The edges in the generated subgraph ", G_gen.edges)

    possible_target_graphs = []

    for idx, graph in enumerate(dataset):
        if graph.y.item() != target_label:
            possible_target_graphs.append(idx)

    np.random.seed(args.seed)
    injected_graph_idx = np.random.choice(
        possible_target_graphs, int(inject_ratio * len(dataset))
    )

    backdoor_dataset = []
    clean_dataset = []
    all_dataset = []

    for idx, graph in enumerate(dataset):
        if idx not in injected_graph_idx:
            all_dataset.append(graph)
            clean_dataset.append(graph)
            continue

        if graph.num_nodes > backdoor_num:
            np.random.seed(args.seed)
            random_select_nodes = np.random.choice(
                graph.num_nodes, backdoor_num, replace=False
            )
        else:
            np.random.seed(args.seed)
            random_select_nodes = np.random.choice(graph.num_nodes, backdoor_num)

        removed_index = []
        ls_edge_index = graph.edge_index.T.numpy().tolist()

        # remove existing edges between the selected nodes
        for row_idx, i in enumerate(random_select_nodes):
            for col_idx, j in enumerate(random_select_nodes):
                if [i, j] in ls_edge_index:
                    removed_index.append(ls_edge_index.index([i, j]))

        removed_index = list(set(removed_index))
        remaining_index = np.arange(0, len(graph.edge_index[0, :]))
        remaining_index = np.delete(remaining_index, removed_index)

        graph.edge_index = graph.edge_index[:, remaining_index]
        if graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr[remaining_index, :]

        # inject subgraph trigger into the clean graph
        for edge in G_gen.edges:
            i, j = random_select_nodes[edge[0]], random_select_nodes[edge[1]]

            # injecting edge
            graph.edge_index = torch.cat(
                (graph.edge_index, torch.LongTensor([[int(i)], [int(j)]])), dim=1
            )
            graph.edge_index = torch.cat(
                (graph.edge_index, torch.LongTensor([[int(j)], [int(i)]])), dim=1
            )
            # padding for the edge attributes matrix
            # if graph.edge_attr is not None:
            #     graph.edge_attr = torch.cat(
            #         (
            #             graph.edge_attr,
            #             torch.unsqueeze(torch.zeros_like(graph.edge_attr[0, :]), 0),
            #         ),
            #         dim=0,
            #     )
            #     graph.edge_attr = torch.cat(
            #         (
            #             graph.edge_attr,
            #             torch.unsqueeze(torch.zeros_like(graph.edge_attr[0, :]), 0),
            #         ),
            #         dim=0,
            #     )
        graph.y = torch.Tensor([target_label]).to(torch.int64)
        backdoor_dataset.append(graph)
        all_dataset.append(graph)

    return all_dataset, list(set(injected_graph_idx)), backdoor_dataset, clean_dataset


def preprocess_dataset(dataset, max_degree=0):
    for graph in dataset:
        graph.y = graph.y.view(-1)

    dataset, max_degree = prepare_dataset_x(dataset, max_degree=max_degree)
    return dataset, max_degree


def prepare_dataset_x(dataset, max_degree=0):
    # if dataset[0].x is None:
    if max_degree == 0:
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
            data.num_nodes = int(torch.max(data.edge_index)) + 1
        max_degree = max_degree + 4  # edit this!!!!!!!!!!!!!!!!!!!!!!!!!1
    else:
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            data.num_nodes = int(torch.max(data.edge_index)) + 1
        max_degree = max_degree
    if max_degree < 10000:
        # dataset.transform = T.OneHotDegree(max_degree)
        for data in dataset:
            degs = degree(data.edge_index[0], dtype=torch.long)
            data.x = F.one_hot(degs, num_classes=max_degree + 1).to(torch.float)
    else:
        deg = torch.cat(degs, dim=0).to(torch.float)
        mean, std = deg.mean().item(), deg.std().item()
        for data in dataset:
            degs = degree(data.edge_index[0], dtype=torch.long)
            data.x = ((degs - mean) / std).view(-1, 1)
    return dataset, max_degree


def motif_trans(motif_idx):
    if motif_idx == "M31":
        motif_adj = [(0, 1), (0, 2)]
    elif motif_idx == "M32":
        motif_adj = [(0, 1), (0, 2), (1, 2)]
    elif motif_idx == "M41":
        motif_adj = [(0, 1), (1, 2), (2, 3)]
    elif motif_idx == "M42":
        motif_adj = [(0, 1), (0, 2), (0, 3)]
    elif motif_idx == "M43":
        motif_adj = [(0, 1), (1, 2), (2, 3), (3, 0)]
    elif motif_idx == "M44":
        motif_adj = [(0, 1), (1, 2), (2, 3), (0, 2)]
    elif motif_idx == "M45":
        motif_adj = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]
    elif motif_idx == "M46":
        motif_adj = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
    else:
        raise ValueError("motif_idx not found!")

    return motif_adj
