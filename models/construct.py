from models.GCN import GCN
from models.GIN import GIN
from models.GAT import GAT
from models.GraphSAGE import GraphSAGE


def model_construct(model_name, num_features, num_classes, num_hidden, device):
    if model_name == "GCN":
        model = GCN(num_features, num_classes, num_hidden).to(device)
    elif model_name == "GIN":
        model = GIN(num_features, num_classes, num_hidden).to(device)
    elif model_name == "GAT":
        model = GAT(num_features, num_classes, num_hidden).to(device)
    elif model_name == "GraphSAGE":
        model = GraphSAGE(num_features, num_classes, num_hidden).to(device)
    else:
        raise ValueError("Invalid model name: {}".format(model_name))
    return model
