import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool as gmp

class GCNNet(torch.nn.Module):
    def __init__(self, n_output=2, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=954, output_dim=128, dropout=0.2):

        super(GCNNet, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # SMILES1 graph branch
        self.n_output = n_output
        self.drug_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug_conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.drug_conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.drug_fc_g1 = torch.nn.Linear(num_features_xd*4, num_features_xd*2)
        self.drug_fc_g2 = torch.nn.Linear(num_features_xd*2, output_dim)

        # DL cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim * 2),
            nn.ReLU()
        )

        # combined layers
        self.fc1 = nn.Linear(output_dim * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, n_output)

    def forward(self, data1, data2):
        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # deal drug1
        x1 = self.drug_conv1(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = self.drug_conv2(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = self.drug_conv3(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1)       # global max pooling

        # flatten
        x1 = self.relu(self.drug_fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.drug_fc_g2(x1)
        x1 = self.dropout(x1)

        # deal drug2
        x2 = self.drug_conv1(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = self.drug_conv2(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = self.drug_conv3(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = gmp(x2, batch2)  # global max pooling

        # flatten
        x2 = self.relu(self.drug_fc_g1(x2))
        x2 = self.dropout(x2)
        x2 = self.drug_fc_g2(x2)
        x2 = self.dropout(x2)

        # deal cell
        cell_vector = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell_vector)

        # concat
        xc = torch.cat((x1, x2, cell_vector), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc3(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out