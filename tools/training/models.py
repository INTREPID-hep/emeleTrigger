# models.py
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool


class GATRegressor(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim=1):
        super(GATRegressor, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim*2, output_dim)

    def forward(self, data):
        x, edge_index, batch, edge_attrib = data.x.float(), data.edge_index, data.batch, data.edge_attr
        '''        # Verificar la presencia de deltaPhi y deltaEta
                if hasattr(data, 'deltaPhi') and hasattr(data, 'deltaEta'):
                    deltaPhi, deltaEta = data.deltaPhi.float(), data.deltaEta.float()
                    edge_attr = torch.stack([deltaPhi, deltaEta], dim=1).float()
                else:
                    edge_attr = None
        '''
        x = F.relu(x)
        x = self.conv1(x, edge_index, edge_attr=edge_attrib)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attrib)
        x = F.relu(x)
        x = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        x = self.fc1(x)
        return x
    
    
from torch_geometric.nn import GCNConv
class GCNRegressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(GCNRegressor, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels*4)
        self.conv2 = GCNConv(hidden_channels*4, hidden_channels*2)
        self.conv3 = GCNConv(hidden_channels*2, hidden_channels*2)
        self.conv4 = GCNConv(hidden_channels*2, hidden_channels*2)

        self.lin1 = torch.nn.Linear(hidden_channels*2, hidden_channels*2)
        self.lin2 = torch.nn.Linear(hidden_channels*2, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin4 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x.float(), data.edge_index, data.edge_weight, data.batch

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        x = global_max_pool(x, data.batch)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x).squeeze(1)
        return x

from torch_geometric.nn import SAGEConv
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels*4)
        self.conv2 = SAGEConv(hidden_channels*4, hidden_channels*2)
        self.conv3 = SAGEConv(hidden_channels*2, hidden_channels*2)
        self.conv4 = SAGEConv(hidden_channels*2, hidden_channels*2)

        self.lin1 = torch.nn.Linear(hidden_channels*2, hidden_channels*2)
        self.lin2 = torch.nn.Linear(hidden_channels*2, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin4 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x).squeeze(1)
       
        return x
    

from torch_geometric.nn import MessagePassing, AttentionalAggregation
from torch_geometric.utils import add_self_loops, degree, softmax

class MPL(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPL, self).__init__(aggr='add')
        self.mlp1 = torch.nn.Linear(in_channels*2, out_channels)
        self.mlp2 = torch.nn.Linear(in_channels, out_channels)
        self.mlp3 = torch.nn.Linear(2*out_channels, 1)
        self.mlp4 = torch.nn.Linear(2*out_channels, 1)
        self.mlp5 = torch.nn.Linear(in_channels,16)
        self.mlp6 = torch.nn.Linear(out_channels,16)
        self.mlp7 = torch.nn.Linear(16,1)

    def forward(self, x, edge_index):
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        msg = self.propagate(edge_index, x=x.float())
        x = F.relu(self.mlp2(x))
        w1 = F.sigmoid(self.mlp3(torch.cat([x,msg], dim=1)))
        w2 = F.sigmoid(self.mlp4(torch.cat([x,msg], dim=1)))
        out = w1*msg + w2*x
        
        return out

    def message(self, x_i, x_j, edge_index):
        msg = F.relu(self.mlp1(torch.cat([x_i, x_j-x_i], dim=1)))
        w1 = F.tanh(self.mlp5(x_i))
        w2 = F.tanh(self.mlp6(msg))
        w = self.mlp7(w1*w2)
        w = softmax(w, edge_index[0])
        return msg*w

class MPLNNRegressor(torch.nn.Module):
    def __init__(self,in_channels):
        super(MPLNNRegressor, self).__init__()
        self.conv1 = MPL(in_channels,128 )
        self.conv2 = MPL(128,64)
        self.conv3 = MPL(64,64 )
        self.conv4 = MPL(64,64 )
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 16)
        self.lin3 = torch.nn.Linear(16, 16)
        self.lin4 = torch.nn.Linear(16, 1)
        self.lin5 = torch.nn.Linear(128, 128)
        self.lin6 = torch.nn.Linear(128, 16)
        self.lin7 = torch.nn.Linear(16, 16)
        self.lin8 = torch.nn.Linear(16, 1)
        self.global_att_pool1 = AttentionalAggregation(torch.nn.Sequential(torch.nn.Linear(64, 1)))
        self.global_att_pool2 = AttentionalAggregation(torch.nn.Sequential(torch.nn.Linear(64, 1)))
    
    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x1 = self.global_att_pool1(x, batch)
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x2 = self.global_att_pool2(x, batch)
        x_out = torch.cat([x1, x2], dim=1)
        x = F.relu(self.lin1(x_out))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        x = self.lin4(x).squeeze(1)

        return x



class GCNNodeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNNodeClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        # Primero se aplican dos capas GCN sin pooling, para preservar la predicción a nivel de nodo.
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        # La capa final devuelve logits para cada nodo (por ejemplo, dos clases: true o fake)
        out = self.conv3(x, edge_index)
        return out

   
class EdgeGNNClassifier(nn.Module):
    def __init__(self, in_channels, edge_in, hidden_channels):
        super().__init__()
        # Encoder que usa los atributos de aristas
        self.conv = TransformerConv(in_channels, hidden_channels, edge_dim=edge_in, heads=4, concat=False)
        # MLP para clasificar aristas
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hidden_channels + edge_in, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        # 1. Obtener embeddings nodales
        x = self.conv(x, edge_index, edge_attr)  # -> [num_nodes, hidden_channels]

        # 2. Recolectar embeddings de origen y destino
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        # 3. Concatenar embeddings + atributos de arista
        edge_feat = torch.cat([x_src, edge_attr, x_dst], dim=-1)

        # 4. Clasificador: logit por arista
        return self.edge_mlp(edge_feat).view(-1)

##### Edge Classifiers #####

class EdgeClassifierGCN(nn.Module):
    def __init__(self, in_channels, edge_in_channels, hidden_dim=64, model_type="SAGE", dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        

        edge_mlp_in = 2 * hidden_dim + edge_in_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
            edge_attr = data.edge_attr.float()
            edge_feat = torch.cat([x_src, x_dst, edge_attr], dim=1)
        else:
            edge_feat = torch.cat([x_src, x_dst], dim=1)

        logits = self.edge_mlp(edge_feat).squeeze(-1)
        return logits


class EdgeClassifierSAGE(nn.Module):
    def __init__(self, in_channels, edge_in_channels, hidden_dim=64, model_type="SAGE", dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)

        edge_mlp_in = 2 * hidden_dim + edge_in_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
            edge_attr = data.edge_attr.float()
            edge_feat = torch.cat([x_src, x_dst, edge_attr], dim=1)
        else:
            edge_feat = torch.cat([x_src, x_dst], dim=1)

        logits = self.edge_mlp(edge_feat).squeeze(-1)
        return logits


class EdgeClassifierTransformerConv(nn.Module):
    def __init__(self, in_channels, edge_in_channels, hidden_dim=64, model_type="SAGE", dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.conv1 = TransformerConv(in_channels, hidden_dim*2, edge_dim=edge_in_channels, heads=4, concat=False)
        self.conv2 = TransformerConv(hidden_channels*2, hidden_dim, edge_dim=edge_in_channels, heads=4, concat=False)
        self.conv3 = TransformerConv(hidden_channels, hidden_dim, edge_dim=edge_in_channels, heads=4, concat=False)

        edge_mlp_in = 2 * hidden_dim + edge_in_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
            edge_attr = data.edge_attr.float()
            edge_feat = torch.cat([x_src, x_dst, edge_attr], dim=1)
        else:
            edge_feat = torch.cat([x_src, x_dst], dim=1)

        logits = self.edge_mlp(edge_feat).squeeze(-1)
        return logits


class EdgeClassifierSAGEUpgrade(nn.Module):
    def __init__(self, in_channels, edge_in_channels, hidden_dim=64, model_type="SAGE", dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.conv1 = SAGEConv(in_channels, hidden_dim*2, edge_dim=edge_in_channels)
        self.conv2 = SAGEConv(hidden_channels*2, hidden_dim, edge_dim=edge_in_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_dim, edge_dim=edge_in_channels)
        
        edge_mlp_in = 2 * hidden_dim + edge_in_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
            edge_attr = data.edge_attr.float()
            edge_feat = torch.cat([x_src, x_dst, edge_attr], dim=1)
        else:
            edge_feat = torch.cat([x_src, x_dst], dim=1)

        logits = self.edge_mlp(edge_feat).squeeze(-1)
        return logits
