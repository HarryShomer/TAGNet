import torch 
from torch_scatter import scatter

import os
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")


class DistDataset:
    """
    Dataset with node and edge constraint
    """
    def __init__(
        self, 
        dataset_name,
        delta,
        device="cuda", 
    ):
        self.delta = delta
        self.device = device 
        self.dataset_name = dataset_name
        

    def get_delta_mask(self, delta, dist_sources_2_nodes, max_iter):
        """
        Get mask corresponding to specific delta.

        We want to return a mask of shape (max_iter, BS, |V|) which is 1 at
        the iteration when it's the 'delta'-th time we visit the node else 0
        """
        all_dists_masks = []

        for dist in range(1, max_iter+1):
            x = (dist_sources_2_nodes - dist) == delta
            all_dists_masks.append(x.unsqueeze(0))

        return torch.cat(all_dists_masks, dim=0)


    def num_non_informative_msgs(self, dist_sources_2_nodes, edge_index, iteration):
        """
        For all edges, check if the head node was reached yet
        """
        source_2_heads = dist_sources_2_nodes[:][:, edge_index[0]]
        head_reached = (source_2_heads >= iteration).to(torch.int)

        num_ni_msgs = scatter(head_reached, edge_index[1], dim=1, dim_size=dist_sources_2_nodes.shape[1], reduce="sum")

        return num_ni_msgs
        


    def only_node_constraint(self, dist_sources_2_nodes, iteration):
        """
        Check the node constraint on *all* nodes
        """
        return (dist_sources_2_nodes >= iteration - self.delta) & (iteration >= dist_sources_2_nodes)


    def get_mask(self, dist_sources_2_nodes, edge_index, iteration):
        """
        For a batch of queries for a single iteration, pull all the edges for each.

        Constraints:
        -----------
        1. Node Constraint: k - \delta <= dist(s, v) <= k 
        2. Edge Constraint: dist(s, u) < dist(s, v) + \delta
                                                                            
        Returns:
        --------
        Torch.Tensor
            Mask for edges to use for each sample in batch
        """
        # Distance from query nodes to all heads and tails
        source_2_heads = dist_sources_2_nodes[:][:, edge_index[0]]
        source_2_tails = dist_sources_2_nodes[:][:, edge_index[1]]

        node_constraint_mask = (source_2_tails >= iteration - self.delta) & (iteration >= source_2_tails)
        edge_constraint_mask = (source_2_heads < source_2_tails + self.delta)

        # Both conditions
        # Shape = (BS, |Edges|)
        both_masks = (edge_constraint_mask & node_constraint_mask)

        return both_masks


    def get_all_masks(self, dist_sources_2_nodes, edge_index, max_layers):
        """
        For a batch of queries for a single iteration, pull all the edges for each.

        Constraints:
        -----------
        1. Node Constraint: k - \delta <= dist(s, v) <= k 
        2. Edge Constraint: dist(s, u) < dist(s, v) + \delta

        Returns:
        --------
        list
            Each mask tensor of edges to include
        """
        # Distance from query nodes to all heads and tails
        source_2_heads = dist_sources_2_nodes[:][:, edge_index[0]]
        source_2_tails = dist_sources_2_nodes[:][:, edge_index[1]]

        # Always same
        edge_constraint_mask = (source_2_heads < source_2_tails + self.delta)

        all_masks = []

        for layer in range(1, max_layers+1):
            node_constraint_mask = (layer >= source_2_tails) & (source_2_tails >= layer - self.delta)
            layer_mask = (edge_constraint_mask & node_constraint_mask) # Shape = (BS, |Edges|)
            all_masks.append(layer_mask.unsqueeze(0))  

        all_masks = torch.cat(all_masks, dim=0)
        all_masks = torch.nonzero(all_masks).T

        return all_masks