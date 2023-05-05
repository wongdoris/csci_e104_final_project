# CSCI-E104 Final Project
## DeepDDS: Deep Graph Neural Network with attention mechanism to predict synergistic drug combinations


This is an implementation of the GNN models shown in the paper [DeepDDS: Deep Graph Neural Network with attention mechanism to predict synergistic drug combinations](https://paperswithcode.com/paper/deepdds-deep-graph-neural-network-with).

Drug combination therapy has become a increasingly promising method in the treatment of cancer. However, the number of possible drug combinations is so huge that it is hard to screen synergistic drug combinations through wet-lab experiments. Therefore, computational screening has become an important way to prioritize drug combinations. Graph neural network have recently shown remarkable performance in the prediction of compound-protein interactions, but it has not been applied to the screening of drug combinations.

This project demonstrates the DeepDDS framework proposed by the paper. The data processing, model building and training codes are streamlined and packaged together in `train_model.py`. A prediction process `predict.py` is added to generate model predictions easily. And on top of the **GCN and GAT** models used in the original paper, two more GNN models are added: **Sage and GIN** to expand the options on building your own Deep GNN framework for synergistic drug combinations prediction.


## To train a model:
```python train_model.py -model gat -epoch 100 -name gat_model1```

The script above trains a GAT model for 100 epochs and save the trained model with name "gat_model1".

The options for `-model` are: `gat, gcn, gin, sage`


## To generate predictions:

```python predict.py -model gat -name gat_model1 -data independent_input```

Please make sure model type aligns with the saved model for the prediction process to work.


## Using Google Colab:
Clone the repository and install requirements:
```
%cd /content/
!git clone https://github.com/wongdoris/csci_e104_final_project.git
%cd /content/csci_e104_final_project
!pip install -r requirements.txt
```

Train a model:

```!python train_model.py -model gat -epoch 100 -name gat_model1```

Generate predictions:

```!python predict.py -model gat -name gat_model1 -data independent_input```