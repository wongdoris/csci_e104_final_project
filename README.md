# CSCI-E104 Final Project
## DeepDDS: Deep Graph Neural Network with attention mechanism to predict synergistic drug combinations


This is an implementation of the GNN models shown in the paper [DeepDDS: Deep Graph Neural Network with attention mechanism to predict synergistic drug combinations](https://paperswithcode.com/paper/deepdds-deep-graph-neural-network-with).

Drug combination therapy has become a increasingly promising method in the treatment of cancer. However, the number of possible drug combinations is so huge that it is hard to screen synergistic drug combinations through wet-lab experiments. Therefore, computational screening has become an important way to prioritize drug combinations. Graph neural network have recently shown remarkable performance in the prediction of compound-protein interactions, but it has not been applied to the screening of drug combinations.

In ths project I re-created the moelling framework created by Jinxian Wang, Xuejun Liu, Siyuan Shen, Lei Deng1, and Hui Liu. The data laod and model training process has been streamlined. A prediction process is added to generate model predictions easily. And on top of the **GCN and GAT** models used in the original paper, two more GNN models are added: **Sage and GIN** to expand the options to build your Deep GNN framework for synergistic drug combinations prediction.


## To train a model:
```python train_model.py -model gat -epoch 100 -name gat_model1```

The command above trains a GAT model for 100 epochs and save the trained model with name "gat_model1".

The options for `-model` are: `gat, gcn, gin, sage`


## To generate predictions:

```python predict.py -model gat -name gat_model1 -data independent_input```

Please make sure model type aligns with the saved model for the prediction process to work.