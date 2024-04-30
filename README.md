# GCleaner: Defending against Backdoor Attack on Graph Neural Networks

A PyTorch implementation of "GCleaner: Defending against Backdoor Attack on Graph Neural Networks"

## Requirements

```
python==3.8.19
torch==2.2.2
torch_geometric==2.5.2
scikit-learn==1.3.0
```

## Quick Start

### Run in AIDS

```sh
python main.py --dataset AIDS --model GIN --split_ratio 0.8 --injection_ratio 0.05 --trigger_ratio 0.2 --target_label 1 --attack_method subgraph --epoch 50
```

