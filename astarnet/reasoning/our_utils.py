import torch



def get_mask(dist_sources_2_nodes, edge_index, iteration, delta):
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

    node_constraint_mask = (source_2_tails >= iteration - delta) & (iteration >= source_2_tails)
    edge_constraint_mask = (source_2_heads < source_2_tails + delta)

    # Both conditions
    # Shape = (BS, |Edges|)
    both_masks = (edge_constraint_mask & node_constraint_mask)

    return both_masks


def get_all_masks(dist_sources_2_nodes, edge_index, max_layers, delta):
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
    edge_constraint_mask = (source_2_heads < source_2_tails + delta)

    all_masks = []

    for layer in range(1, max_layers+1):
        node_constraint_mask = (layer >= source_2_tails) & (source_2_tails >= layer - delta)
        layer_mask = (edge_constraint_mask & node_constraint_mask) # Shape = (BS, |Edges|)
        
        # Flatten batch dim
        layer_mask = layer_mask.flatten()
        
        all_masks.append(layer_mask.unsqueeze(0))  

    all_masks = torch.cat(all_masks, dim=0)
    # all_masks = torch.nonzero(all_masks).T

    return all_masks



def bfs(node_idx, num_hops, edge_index, num_nodes):
    """
    Adapted from torch_geometric.utils.k_hop_subgraph function
    """
    col, row = edge_index  # col=head, row=tail

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    node_idx = torch.tensor([node_idx], device=row.device).flatten()

    all_hop_nodes = []
    node_dist_2_all = torch.full((num_nodes,), 100).to(row.device)  # Init distance to 100 for all

    # +1 To account for first pass being dist=0
    for hop in range(num_hops+1):
        node_mask.fill_(False)
        mask_to_apply = node_idx if hop == 0 else col[edge_mask]
        node_mask[mask_to_apply] = True

        # Ignore on last iter since not needed
        if hop != num_hops:
            torch.index_select(node_mask, 0, row, out=edge_mask)  # NOTE: Assigns to edge_mask!!!
        
        if hop != 0:
            hop_nodes = torch.nonzero(node_mask).squeeze(1)
            all_hop_nodes.append(hop_nodes)

    # Start from back because we sometimes get repeats (i.e. node shows up in hop 2 and 3)
    for ix, hop_nodes in enumerate(all_hop_nodes[::-1]):
        node_dist_2_all[hop_nodes] = num_hops - ix
    
    # Source is always 0
    node_dist_2_all[node_idx] = 0

    return node_dist_2_all


def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

def index_to_mask2(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[torch.arange(mask.shape[0]), index] = True
    return mask.to(index.device)