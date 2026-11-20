import torch, numpy as np
import torch_geometric, torch_scatter

from torch.nn import *
from torch_geometric.nn import MessagePassing, GlobalAttention
from torch_geometric.data import Data, Batch

def get_start_indices(splits):
    splits = torch.roll(splits, 1)
    splits[0] = 0

    start_indices = torch.cumsum(splits, 0)
    return start_indices

def masked_segmented_softmax(energies, mask, start_indices, batch_ind):
    mask = mask + start_indices
    mask_bool = torch.ones_like(energies, dtype=torch.bool) # inverse mask matrix
    mask_bool[mask] = False

    energies[mask_bool] = -np.inf
    probs = torch_geometric.utils.softmax(energies, batch_ind) # to probs ; per graph

    return probs

def segmented_sample(probs, splits):
    probs_split = torch.split(probs, splits)
    samples = [torch.multinomial(x, 1) for x in probs_split]
    
    return torch.cat(samples)

def segmented_scatter_(dest, indices, start_indices, values):
    real_indices = start_indices + indices
    dest[real_indices] = values
    return dest

def segmented_gather(src, indices, start_indices):
    real_indices = start_indices + indices
    return src[real_indices]

EMB_SIZE = 32
# ----------------------------------------------------------------------------------------
class MultiMessagePassing(Module):
    def __init__(self, steps, node_in_size=EMB_SIZE, node_out_size=EMB_SIZE, edge_size=2, global_size=EMB_SIZE, agg_size=EMB_SIZE,activation_fn=LeakyReLU):
        super().__init__()

        # if node_in_size is None:
        #     node_in_size = [EMB_SIZE] * size
            

        self.gnns = ModuleList( [GraphNet(node_in_size, edge_size, global_size, agg_size, node_out_size,activation_fn) for i in range(steps)] )
        self.pools = ModuleList( [GlobalNode(node_out_size, global_size,activation_fn) for i in range(steps)] )

        self.steps = steps

    def forward(self, x, x_global, edge_attr, edge_index, batch_ind, num_graphs):
        for i in range(self.steps):
            x = self.gnns[i](x, edge_attr, edge_index, x_global, batch_ind)
            x_global = self.pools[i](x_global, x, batch_ind)

        return x, x_global

# ----------------------------------------------------------------------------------------
class GlobalNode(Module):       
    def __init__(self, node_size, global_size,activation_fn):
        super().__init__()

        att_mask = Linear(node_size, 1)
        att_feat = Sequential( Linear(node_size, node_size), activation_fn() )

        self.glob = GlobalAttention(att_mask, att_feat)
        self.tranform = Sequential( Linear(global_size * 2, global_size), activation_fn() )

    def forward(self, xg_old, x, batch):
        xg = self.glob(x, batch)

        xg = torch.cat([xg, xg_old], dim=1)
        xg = self.tranform(xg) + xg_old # skip connection

        return xg

# ----------------------------------------------------------------------------------------
class GraphNet(MessagePassing):
    def __init__(self, node_in_size, edge_size, global_size, agg_size, node_out_size,activation_fn):
        super().__init__(aggr='max')

        self.f_mess = Sequential( Linear(node_in_size + edge_size, agg_size), activation_fn() )
        self.f_agg  = Sequential( Linear(node_in_size + global_size + agg_size, node_out_size), activation_fn() )

    def forward(self, x, edge_attr, edge_index, xg, batch):
        xg = xg[batch]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, xg=xg)

    def message(self, x_j, edge_attr):
        z = torch.cat([x_j, edge_attr], dim=1)
        z = self.f_mess(z)

        return z 

    def update(self, aggr_out, x, xg):
        z = torch.cat([x, xg, aggr_out], dim=1)
        z = self.f_agg(z) + x # skip connection

        return z

# ----------------------------------------------------------------------------------------
# EGr (Horcik et al. 2025, Sec. 2): one message function per edge type, no edge labels;
# each type's messages are aggregated separately (max, as GraphNet), and all the
# aggregates go into the combine function together:
#   v' = f_agg([v, x_global, agg_1(v), ..., agg_T(v)]) + v
# Hidden size and skip connection as in GraphNet. Edges come from
# sage.agent.edge_modes.split_by_type.
class TypedGraphNet(Module):
    def __init__(self, node_in_size, num_types, global_size, agg_size, node_out_size, activation_fn):
        super().__init__()

        self.num_types = num_types
        self.f_mess = ModuleList( [Sequential( Linear(node_in_size, agg_size), activation_fn() ) for t in range(num_types)] )
        self.f_agg  = Sequential( Linear(node_in_size + global_size + num_types * agg_size, node_out_size), activation_fn() )

    def forward(self, x, typed_edges, xg, batch):
        """:param typed_edges: list of (src, dst) index tensors, one per type"""
        assert len(typed_edges) == self.num_types
        n = x.shape[0]
        aggs = []
        for f_mess, (src, dst) in zip(self.f_mess, typed_edges):
            m = f_mess(x[src])
            # max over each node's messages of this type; a node with none gets 0
            aggs.append(torch_scatter.scatter(m, dst, dim=0, dim_size=n, reduce='max'))

        z = torch.cat([x, xg[batch]] + aggs, dim=1)
        z = self.f_agg(z) + x # skip connection

        return z

class TypedMultiMessagePassing(Module):
    """MultiMessagePassing with TypedGraphNet layers (EGr); same global node."""
    def __init__(self, steps, num_types, node_in_size=EMB_SIZE, node_out_size=EMB_SIZE, global_size=EMB_SIZE, agg_size=EMB_SIZE, activation_fn=LeakyReLU):
        super().__init__()

        self.gnns = ModuleList( [TypedGraphNet(node_in_size, num_types, global_size, agg_size, node_out_size, activation_fn) for i in range(steps)] )
        self.pools = ModuleList( [GlobalNode(node_out_size, global_size, activation_fn) for i in range(steps)] )

        self.steps = steps

    def forward(self, x, x_global, typed_edges, batch_ind, num_graphs):
        for i in range(self.steps):
            x = self.gnns[i](x, typed_edges, x_global, batch_ind)
            x_global = self.pools[i](x_global, x, batch_ind)

        return x, x_global
 