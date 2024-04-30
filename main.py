import argparse
import numpy as np
import torch
import random
import copy
import os
import pickle
from attack import *
from torch_geometric.loader import DataLoader
from models.construct import model_construct
from utils import *
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
# Training settings
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
)
parser.add_argument("--seed", type=int, default=10, help="Random seed.")
parser.add_argument(
    "--dataset",
    type=str,
    default="AIDS",
    help="Dataset",
    choices=["AIDS", "MUTAG", "NCI1", "COLLAB", "bitcoin", "Fingerprint", "PROTEINS"],
)
parser.add_argument("--num_hidden", type=int, default=64)
parser.add_argument("--epoch", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.005)
parser.add_argument(
    "--split_ratio", default=0.8, type=float, help="train/test split ratio"
)
# Backdoor settings
parser.add_argument(
    "--model",
    type=str,
    default="GIN",
    choices=["GCN", "GIN", "GAT", "GraphSAGE"],
    help="Model used to attack",
)
parser.add_argument(
    "--attack_method",
    default="subgraph",
    type=str,
    choices=["subgraph", "motif", "GTA", "exp"],
    help="Method used to attack",
)
parser.add_argument(
    "--injection_ratio",
    default=0.05,
    type=float,
    help="the number of injected samples to the training dataset",
)
parser.add_argument(
    "--trigger_ratio",
    default=0.2,
    type=float,
    help="Ratio of trigger nodes to average nodes",
)
parser.add_argument(
    "--trigger_density", default=0.8, type=float, help="Density of Subgraph Triggers"
)
parser.add_argument("--target_label", default=1, type=int, help="Label to be attacked")
parser.add_argument(
    "--motif_name", default="M44", type=str, help="Name of motif trigger"
)
parser.add_argument(
    "--gtn_layernum", type=int, default=3, help="layer number of GraphTrojanNet"
)
parser.add_argument("--gtn_lr", type=float, default=0.01)
parser.add_argument("--gtn_epochs", type=int, default=20, help="# attack epochs")
parser.add_argument(
    "--topo_thrd", type=float, default=0.5, help="threshold for topology generator"
)
parser.add_argument(
    "--topo_activation",
    type=str,
    default="sigmoid",
    help="activation function for topology generator",
)
# Unlearning settings
parser.add_argument(
    "--un_split_ratio",
    default=0.25,
    type=float,
    help="unlearning train/test split ratio",
)
parser.add_argument("--un_epoch", type=int, default=100)
parser.add_argument("--un_learning_rate", type=float, default=0.001)
parser.add_argument("--un_log_dir", type=str, default="myresults")
# defense setting
parser.add_argument(
    "--defense_mode",
    type=str,
    default="unlearning",
    choices=["prune", "unlearning", "ABL", "RS"],
    help="Mode of defense",
)
parser.add_argument(
    "--prune_thr", type=float, default=0.1, help="Threshold of prunning edges"
)
parser.add_argument(
    "--debug_print", action="store_true", help="Print debug information"
)
# GPU setting
parser.add_argument("--device_id", type=int, default=0, help="GPU device id")
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(("cuda:{}" if args.cuda else "cpu").format(args.device_id))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)
file_name = "save_model/{}_{}_ratio_{}_triggerratio_{}_{}_{}.pkl".format(
    args.dataset,
    args.model,
    args.injection_ratio,
    args.trigger_ratio,
    args.target_label,
    args.attack_method,
)
attr_name = file_name.replace(".pkl", "_attr.txt")
clean_name = file_name.replace(".pkl", "_clean.pth")

# trian a backdoor model
if os.path.exists(file_name) == False:
    dataset, num_classes = load_data(args.dataset)
    avg_num_nodes, _, _, _, _, _, max_num_nodes = stat_graph(dataset)
    print("# Average Number of Nodes: {}".format(avg_num_nodes))
    trigger_size = int(avg_num_nodes * args.trigger_ratio)
    if trigger_size < 2:
        trigger_size = 2
    print("# Trigger Size: {}".format(trigger_size))

    random.Random(args.seed).shuffle(dataset)
    train_nums = int(len(dataset) * args.split_ratio)
    train_dataset = dataset[:train_nums]
    test_dataset = dataset[train_nums:]

    if args.attack_method == "subgraph":
        (
            train_dataset,
            injected_graph_idx,
            backdoor_train_dataset,
            clean_train_dataset,
        ) = inject_sub_trigger(
            args,
            copy.deepcopy(train_dataset),
            inject_ratio=args.injection_ratio,
            target_label=args.target_label,
            backdoor_num=trigger_size,
            density=args.trigger_density,
        )

        _, _, backdoor_test_dataset, _ = inject_sub_trigger(
            args,
            copy.deepcopy(test_dataset),
            inject_ratio=1,
            target_label=args.target_label,
            density=args.trigger_density,
            backdoor_num=trigger_size,
        )

        _, _, _, clean_test_dataset = inject_sub_trigger(
            args,
            copy.deepcopy(test_dataset),
            inject_ratio=0,
            target_label=args.target_label,
        )
    else:
        raise NotImplementedError

    print("# Train Dataset {}".format(len(train_dataset)))
    print("# Backdoor Train Dataset {}".format(len(backdoor_train_dataset)))
    print("# Clean Train Dataset {}".format(len(clean_train_dataset)))

    print("# Test Dataset {}".format(len(test_dataset)))
    print("# Backdoor Test Dataset {}".format(len(backdoor_test_dataset)))
    print("# Clean Test Dataset {}".format(len(clean_test_dataset)))

    train_dataset, max_degree = preprocess_dataset(train_dataset)
    if os.path.exists("save_model") == False:
        os.mkdir("save_model")
    with open(attr_name, "w") as f:
        f.write("{}\n".format(max_degree))

    backdoor_test_dataset, _ = preprocess_dataset(
        backdoor_test_dataset, max_degree=max_degree
    )
    clean_test_dataset, _ = preprocess_dataset(
        clean_test_dataset, max_degree=max_degree
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=args.batch_size)
    backdoor_test_loader = DataLoader(backdoor_test_dataset, batch_size=args.batch_size)

    model = model_construct(
        args.model, train_dataset[0].x.shape[1], num_classes, args.num_hidden, device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epoch + 1):
        train_loss = trian(model, optimizer, criterion, train_loader, device)

        train_acc, _ = test(model, criterion, train_loader, device)
        test_acc, test_loss = test(model, criterion, clean_test_loader, device)
        backdoor_test_acc, backdoor_test_loss = test(
            model, criterion, backdoor_test_loader, device
        )

        scheduler.step()
        if epoch % 5 == 0:
            print(("=================Epoch {}=================".format(epoch)))
            print("Train Loss: {:.6f}".format(train_loss))
            print("[Trainset] Train Accuracy: {:.6f}".format(train_acc))
            print("[Clean Test] Test Accuracy: {:.6f}".format(test_acc))
            print("[Backdoor Test] Test Accuracy: {:.6f}".format(backdoor_test_acc))

    pickle.dump(model, open(file_name, "wb"))
else:
    model = pickle.load(open(file_name, "rb"))

if args.defense_mode != "unlearning":
    raise NotImplementedError

print("=================Unlearning=================")

dataset, num_classes = load_data(args.dataset)
data_len = len(dataset)
avg_num_nodes, _, _, _, _, _, max_num_nodes = stat_graph(dataset)
print("# Average Number of Nodes: {}".format(avg_num_nodes))
trigger_size = int(avg_num_nodes * args.trigger_ratio)
if trigger_size < 2:
    trigger_size = 2
print("# Trigger Size: {}".format(trigger_size))

random.Random(args.seed).shuffle(dataset)
train_nums = int(len(dataset) * args.split_ratio)
dataset = dataset[train_nums:]
train_nums = int(len(dataset) * args.un_split_ratio)
train_dataset = dataset[:train_nums]
test_dataset = dataset[train_nums:]


if args.attack_method == "subgraph":
    _, injected_graph_idx, backdoor_train_dataset, _ = inject_sub_trigger(
        args,
        copy.deepcopy(train_dataset),
        inject_ratio=1,
        target_label=args.target_label,
        backdoor_num=trigger_size,
        density=args.trigger_density,
    )
    _, _, backdoor_test_dataset, _ = inject_sub_trigger(
        args,
        copy.deepcopy(test_dataset),
        inject_ratio=1,
        target_label=args.target_label,
        density=args.trigger_density,
        backdoor_num=trigger_size,
    )
    _, _, _, clean_test_dataset = inject_sub_trigger(
        args,
        copy.deepcopy(test_dataset),
        inject_ratio=0,
        target_label=args.target_label,
    )
else:
    raise NotImplementedError
clean_train_dataset = [train_dataset[i] for i in injected_graph_idx]
print("# Trian Dataset {}".format(len(clean_train_dataset)))
print("Unlearning Trian num {:.2f}".format(len(backdoor_train_dataset) / data_len))
print("# Test Dataset {}".format(len(test_dataset)))
print("# Backdoor Test Dataset {}".format(len(backdoor_test_dataset)))
print("# Clean Test Dataset {}".format(len(clean_test_dataset)))

if os.path.exists(attr_name):
    with open(attr_name, "r") as f:
        max_degree = int(f.readline())
else:
    print("Warning: No attr file, code may report errors")
    max_degree = 0

backdoor_train_dataset, _ = preprocess_dataset(backdoor_train_dataset, max_degree)
clean_train_dataset, _ = preprocess_dataset(clean_train_dataset, max_degree)
backdoor_test_dataset, _ = preprocess_dataset(backdoor_test_dataset, max_degree)
clean_test_dataset, _ = preprocess_dataset(clean_test_dataset, max_degree)

clean_train_loader = DataLoader(clean_train_dataset, batch_size=args.batch_size)
backdoor_train_loader = DataLoader(backdoor_train_dataset, batch_size=args.batch_size)
clean_test_loader = DataLoader(clean_test_dataset, batch_size=args.batch_size)
backdoor_test_loader = DataLoader(backdoor_test_dataset, batch_size=args.batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=args.un_learning_rate)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()

static_model = pickle.load(open(file_name, "rb"))

if os.path.exists(args.un_log_dir) == False:
    os.mkdir(args.un_log_dir)

log_name = (
    args.un_log_dir
    + "/{}_{}_{}_ratio{}_triggerratio{}_unsplitratio{}_unlr{}_{}_{}.csv".format(
        args.dataset,
        args.model,
        args.attack_method,
        args.injection_ratio,
        args.trigger_ratio,
        args.un_split_ratio,
        args.un_learning_rate,
        args.defense_mode,
        args.target_label,
    )
)
f = open(log_name, "w")
f.write("epoch,train_loss,clean_test_acc,backdoor_test_acc\n")

test_acc, test_loss = test(model, criterion, clean_test_loader, device)
backdoor_test_acc, backdoor_test_loss = test(
    model, criterion, backdoor_test_loader, device
)
f.write("{},-,{},{}\n".format(0, test_acc, backdoor_test_acc))

for epoch in range(1, args.un_epoch + 1):
    model.train()
    loss_all = 0
    graph_all = 0
    for clean_data, backdoor_data in zip(clean_train_loader, backdoor_train_loader):
        optimizer.zero_grad()
        model.zero_grad()
        clean_data = clean_data.to(device)
        backdoor_data = backdoor_data.to(device)

        output_m_c = model(clean_data.x, clean_data.edge_index, clean_data.batch)
        loss_attention = criterion(output_m_c, clean_data.y)
        loss_attention.backward(retain_graph=True)
        final_conv_acts = model.final_conv_acts
        final_conv_grads = model.final_conv_grads
        clean_grad_cam_weights = grad_cam(final_conv_acts, final_conv_grads)
        clean_grad_cam_weights = (
            MinMaxScaler(feature_range=(0, 1))
            .fit_transform(np.array(clean_grad_cam_weights).reshape(-1, 1))
            .reshape(
                -1,
            )
        )
        clean_grad_cam_weights = torch.Tensor(clean_grad_cam_weights)

        output_m_b = model(
            backdoor_data.x, backdoor_data.edge_index, backdoor_data.batch
        )
        loss_attention = criterion(output_m_b, clean_data.y)
        loss_attention.backward(retain_graph=True)
        final_conv_acts = model.final_conv_acts
        final_conv_grads = model.final_conv_grads
        backdoor_grad_cam_weights = grad_cam(final_conv_acts, final_conv_grads)
        backdoor_grad_cam_weights = (
            MinMaxScaler(feature_range=(0, 1))
            .fit_transform(np.array(backdoor_grad_cam_weights).reshape(-1, 1))
            .reshape(
                -1,
            )
        )
        backdoor_grad_cam_weights = torch.Tensor(backdoor_grad_cam_weights)

        output_s_c = static_model(clean_data.x, clean_data.edge_index, clean_data.batch)

        optimizer.zero_grad()
        model.zero_grad()

        loss1 = torch.norm(output_s_c - output_m_c)
        loss2 = torch.norm(output_m_c - output_m_b)
        loss3 = torch.norm(clean_grad_cam_weights - backdoor_grad_cam_weights)
        loss = loss1 + loss2 + loss3
        if args.debug_print:
            print(
                "loss1: {:.6f}, loss2: {:.6f}, loss3: {:.6f}".format(
                    loss1, loss2, loss3
                )
            )
        loss.backward()

        loss_all += loss.item() * clean_data.num_graphs
        graph_all += clean_data.num_graphs
        optimizer.step()

    loss = loss_all / graph_all
    scheduler.step()

    test_acc, test_loss = test(model, criterion, clean_test_loader, device)
    backdoor_test_acc, backdoor_test_loss = test(
        model, criterion, backdoor_test_loader, device
    )

    print(("=================Epoch {}=================".format(epoch)))
    print("Train Loss: {:.6f}".format(loss))
    print("[Clean Test] Test Accuracy: {:.6f}".format(test_acc))
    print("[Backdoor Test] Test Accuracy: {:.6f}".format(backdoor_test_acc))
    f.write("{},{},{},{}\n".format(epoch, loss, test_acc, backdoor_test_acc))

f.close()
