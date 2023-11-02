import torch


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