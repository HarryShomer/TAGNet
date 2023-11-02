import dgl
import copy
import torch
from torch import nn
from collections import defaultdict
from collections.abc import Sequence
from torch.nn import functional as F

from . import tasks
from .layers import LayerConv
from .model_utils import *



class TAGNet(nn.Module):

    def __init__(
            self, 
            input_dim, 
            hidden_dims, 
            num_relation, 
            num_nodes,
            aggregate_func="pna", 
            activation="relu", 
            dependent=True,
            remove_one_hop=False, 
            boundary_condition=False,
            dist_dataset=None, 
            bias=True, 
            edge_index=None,
            dgl_graph=None, 
            train_degree=None, 
            test_degree=None, 
            drop_edge=False,  
            drop=False, 
            weight_delta=False,
            num_heads=1,
            combine_heads="concat",
            att_type="diff",
            att_drop=0,
            temp=1,
            export_att=False,
            degree_msgs=False,
            **kwargs
        ):
        super(TAGNet, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.num_nodes = num_nodes
        self.remove_one_hop = remove_one_hop  # whether to dynamically remove one-hop edges from edge_index
        self.dist_dataset = dist_dataset
        self.dgl_graph = dgl_graph
        self.weight_delta = weight_delta
        self.max_train_node = self.max_node_in_index(edge_index)
        self.export_att = export_att

        self.test_degree = test_degree 
        self.train_degree = train_degree

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(LayerConv(i+1, self.dims[i], self.dims[i + 1], num_relation,
                                         self.dims[0], aggregate_func, activation, 
                                         dependent, bias, drop_edge, drop, 
                                         boundary_condition, degree_msgs))

        feature_dim = hidden_dims[-1] + input_dim

        # additional relation embedding which serves as an initial 'query' for the NBFNet forward pass
        # each layer has its own learnable relations matrix, so we send the total number of relations, too
        self.query = nn.Embedding(num_relation, input_dim)


        if weight_delta:
            self.temp=temp
            self.att_drop = att_drop
            self.att_type = att_type.lower()
            self.num_heads = num_heads
            self.combine_heads = combine_heads.lower()

            # For score we just use the score function
            if self.att_type != "score":
                self.att_mlps = nn.ModuleList([self.create_att_module(input_dim * 2) for _ in range(num_heads)])
            
            # TODO: ???
            if num_heads > 1:
                self.att_head_projs = nn.ModuleList([nn.Linear(input_dim * 2, input_dim * 2) for _ in range(num_heads)])

        # Score function        
        if weight_delta and self.combine_heads == "concat":
            self.mlp = self.create_mlp(input_dim * (num_heads + 1), 1)  # +1 is for query relation
        else:
            self.mlp = self.create_mlp(feature_dim, 1)


    def max_node_in_index(self, edge_index):
        return max(edge_index[0].max().item(), edge_index[1].max().item())



    def create_att_module(self, input_dim):
        """
        Follows GAT design
        """
        mlp = []

        mlp.append(nn.Linear(input_dim, input_dim))
        mlp.append(nn.Linear(input_dim, 1))
        mlp.append(nn.LeakyReLU(negative_slope=0.2))
        
        return nn.Sequential(*mlp)
        
        
    def create_mlp(self, in_dim, out_dim):
        """
        Create a 2 layer MLP 

        Parameters:
        ----------
            in_dim: int 
                Input dimension (also use as hidden dim)
            out_dim: int
                Output dimension

        Returns:
        --------
            torch.nn.sequential object
        """
        mlp = []

        mlp.append(nn.Linear(in_dim, in_dim))
        mlp.append(nn.ReLU()) 
        mlp.append(nn.Linear(in_dim, out_dim))
        
        return nn.Sequential(*mlp)


    def remove_easy_edges(self, data, h_index, t_index, r_index=None):
        # we remove training edges (we need to predict them at training time) from the edge index
        # think of it as a dynamic edge dropout
        h_index_ext = torch.cat([h_index, t_index], dim=-1)
        t_index_ext = torch.cat([t_index, h_index], dim=-1)
        r_index_ext = torch.cat([r_index, r_index + self.num_relation // 2], dim=-1)

        if self.remove_one_hop:
            # we remove all existing immediate edges between heads and tails in the batch
            edge_index = data.edge_index
            easy_edge = torch.stack([h_index_ext, t_index_ext]).flatten(1)
            remove_edge_index = tasks.edge_match(edge_index, easy_edge)[0]
            mask = ~index_to_mask(remove_edge_index, data.num_edges)
        else:
            # we remove existing immediate edges between heads and tails in the batch with the given relation
            edge_index = torch.cat([data.edge_index, data.edge_type.unsqueeze(0)])
            # note that here we add relation types r_index_ext to the matching query
            easy_edge = torch.stack([h_index_ext, t_index_ext, r_index_ext]).flatten(1)
            remove_edge_index = tasks.edge_match(edge_index, easy_edge)[0]
            mask = ~index_to_mask(remove_edge_index, data.num_edges)

        dgl_graph_remove = dgl.remove_edges(self.dgl_graph, remove_edge_index)

        # Removal from data object
        data = copy.copy(data)
        data.edge_index = data.edge_index[:, mask]
        data.edge_type = data.edge_type[mask]

        return data, dgl_graph_remove


    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation // 2)
        return new_h_index, new_t_index, new_r_index


    def batched_bfs(self, source_nodes, num_nodes, edge_index, testing):
        """
        For all source nodes, calculate the shortest path distance to all other nodes.

        Each "node_bfs" holds a list of list where elements in [i] corresponds to nodes i hops away

        if source node is a node not in train_data...will keep searching and eventually run out of memory...so we control for that
        """
        all_nodes_bfs = []

        for node in source_nodes.tolist():
            # NOTE: Only applies to WN18RR where some nodes aren't in the training graph
            # NOTE: Be careful!!! Not done in inductive setting(2nd condition)
            if node > self.max_train_node and not testing:
                # We only know the distance to itself so set everything except source to 100
                node_bfs = torch.full((num_nodes,), 100).to(edge_index.device)
                node_bfs[node] = 0
            else:
                node_bfs = bfs(node, len(self.layers), edge_index, num_nodes)
            
            # node_bfs = torch.ones_like(node_bfs).to(node_bfs.device)  # DEBUGGING ONLY!!!
            all_nodes_bfs.append(node_bfs)

        return torch.stack(all_nodes_bfs)


    def prop_batch(self, data, h_index, r_index, t_index=None, true_ent=None, testing=False):
        """
        Propagate batch
        """        
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query(r_index)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)

        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)

        degree_agg = self.test_degree if testing else self.train_degree

        # Reshaping b4 everything for more efficient message passing
        boundary = boundary.reshape(-1, self.dims[0])
        layer_input = boundary

        dist_sources_2_nodes = self.batched_bfs(h_index, data.num_nodes, data.edge_index, testing) 
        all_layer_masks = self.dist_dataset.get_all_masks(dist_sources_2_nodes, data.edge_index, len(self.layers))

        if not self.training:
            self.batch_dist_2_tails = dist_sources_2_nodes

        hiddens = []

        for layer_num, layer in enumerate(self.layers):
            layer_mask = all_layer_masks[:, all_layer_masks[0] == layer_num]
            layer_mask = layer_mask[1:]  # Remove dim specifying the layer #

            ni_msgs = self.dist_dataset.num_non_informative_msgs(dist_sources_2_nodes, data.edge_index, layer_num+1)

            hidden = layer(layer_input, query, data.edge_index, data.edge_type, size, boundary, layer_mask, degree_agg, ni_msgs)

            hidden = hidden + layer_input  # Residual connection
            hiddens.append(hidden.reshape(batch_size, data.num_nodes, -1))
            layer_input = hidden

        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)  

        if self.weight_delta:
            hidden = self.score_specific_multiple_heads(dist_sources_2_nodes, hiddens, node_query, t_index, true_ent) 

            # Redefine for simplicity
            node_query = query.unsqueeze(1).expand(-1, hidden.shape[1], -1) 
            output = torch.cat([hidden, node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

            # Extract positive sample + negative
            index = t_index.unsqueeze(-1).expand(-1, -1, output.shape[-1])
            output = output.gather(1, index)

        return output


    def forward(self, data, batch, true_ent=None, testing=False):
        h_index, t_index, r_index = batch.unbind(-1)
        shape = h_index.shape

        if self.training:
            data, _ = self.remove_easy_edges(data, h_index, t_index, r_index)

        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)

        feature = self.prop_batch(data, h_index[:, 0], r_index[:, 0], t_index=t_index, true_ent=true_ent, testing=testing)
        score = self.mlp(feature).squeeze(-1)

        return score.view(shape)


    def score_specific_multiple_heads(self, dist_sources_2_nodes, hiddens, node_query, t_index, true_ent):
        """
        Awful
        """
        head_representations = []

        for ix in range(self.num_heads):
            h = self.score_specific(dist_sources_2_nodes, hiddens, node_query, t_index, ix, true_ent)
            head_representations.append(h)
        
        if self.combine_heads == "concat":
            head_representations = torch.cat(head_representations, dim=-1)
        else:
            head_representations = torch.stack(head_representations).mean(dim=0)
                
        return head_representations


    def score_specific(self, dist_sources_2_nodes, hiddens, node_query, t_index, head_num=0, true_ent=None):
        """
        """
        batch_size, num_nodes = dist_sources_2_nodes.shape[0], t_index.shape[1]

        raw_att_scores = []
        for it in range(len(hiddens)):
            it_feature = torch.cat([hiddens[it], node_query], dim=-1)
            node_mask = self.dist_dataset.only_node_constraint(dist_sources_2_nodes, it+1)
            
            # Extract positive sample + negative representations
            index = t_index.unsqueeze(-1).expand(-1, -1, it_feature.shape[-1])
            it_feature = it_feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim) 

            # Do same for the mask
            node_mask = node_mask.gather(1, t_index)
            
            if self.att_type == "score":
                # Project feature first if we are using multiple heads
                it_feature = it_feature if self.num_heads == 1 else self.att_head_projs[head_num](it_feature)
                it_att_out = self.mlp(it_feature).squeeze(-1)
            else:
                it_att_out = self.att_mlps[head_num](it_feature).squeeze(-1)

            it_att_out = (it_att_out * node_mask).unsqueeze(0)

            # NOTE: Mask out by making very small
            # Passing through softmax makes it ~0
            it_att_out = torch.where(it_att_out == 0, torch.tensor(-1e5, dtype=it_att_out.dtype).to(it_att_out.device), it_att_out)

            raw_att_scores.append(it_att_out)

        num_hiddens = len(hiddens)

        raw_att_scores = torch.cat(raw_att_scores, dim=0)
        raw_att_scores = raw_att_scores.reshape(num_hiddens, batch_size * num_nodes)

        att_scores = F.softmax(raw_att_scores / self.temp, dim=0)
        att_scores = att_scores.reshape(num_hiddens, batch_size, num_nodes)

        # Dropout is applied on the scores
        # Consistent with torch geometric implementation for GATConv and GATv2Conv
        if self.training and self.att_drop > 0:
            att_scores = F.dropout(att_scores, p=self.att_drop)


        ##############################
        if self.export_att and not self.training:
            att_scores_true = att_scores[:, torch.arange(batch_size).to(att_scores.device), true_ent]
            
            true_dist = dist_sources_2_nodes[torch.arange(batch_size).to(att_scores.device), true_ent]
            # true_dist = (true_dist - 1).tolist()

            with open(f"att_weights_{self.dist_dataset.dataset_name}_delta-{self.dist_dataset.delta}_layers-{len(self.layers)}.csv", "a") as f:
                for bs, sample_dist in zip(range(batch_size), true_dist):
                    f.write(f"{sample_dist}, " + ", ".join([str(a) for a in att_scores_true[:, bs].tolist()]) + "\n")
                    # print(sample_dist, att_scores_true[:, bs].tolist())
        ##############################

        # Extract positive sample + negative representations for each again...
        index = t_index.unsqueeze(-1).expand(-1, -1, hiddens[0].shape[-1])
        hiddens = [hhh.gather(1, index) for hhh in hiddens]

        final_hid = att_scores[0].unsqueeze(-1) * hiddens[0]
        for it in range(1, len(hiddens)):
            final_hid += att_scores[it].unsqueeze(-1) * hiddens[it]

        return final_hid





