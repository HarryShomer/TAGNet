import math
import torch
from torch import nn

from torch_scatter import scatter
from torch.nn import functional as F

from collections import defaultdict


class LayerConv(nn.Module):

    ALL_TIMES = defaultdict(list)

    def __init__(
            self, 
            layer_num,
            input_dim, 
            output_dim, 
            num_relation, 
            query_input_dim, 
            aggregate_func="pna", 
            activation="relu", 
            dependent=True, 
            bias=True, 
            drop_edge=None, 
            drop=None,
            boundary_condition=False,
            degree_msgs=False,
        ):
        super(LayerConv, self).__init__()
        self.layer_num = layer_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.boundary_condition = boundary_condition
        self.agg_degree_msgs = degree_msgs

        self.agg_func = self.aggregate_with_boundary if boundary_condition else self.aggregate_wo_boundary

        self.drop = drop
        self.drop_edge = drop_edge
        self.layer_norm = nn.LayerNorm(output_dim, elementwise_affine=bias) 

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim, bias=bias)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim, bias=bias)

        if dependent:
            # obtain relation embeddings as a projection of the query relation
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            # relation embeddings as an independent embedding matrix per each layer
            self.relation = nn.Embedding(num_relation, input_dim)

        if degree_msgs:
            self.degree_bias = nn.Parameter(torch.zeros(input_dim))
            stdv = 1. / math.sqrt(self.linear.weight.size(1))
            self.degree_bias.data.uniform_(-stdv, stdv)


    def forward(self, input, query, edge_index, edge_type, adj_size, boundary, mask_ix=None, degree_in=None, ni_msgs=None):
        """
        Input is already reshaped to (BS * |E|, Dim) when passed to function
        """
        num_ents = adj_size[0]
        batch_size = len(query)
        mask_batch_num = mask_ix[0]

        all_rel_ix = torch.index_select(edge_type, 0, mask_ix[1])
        all_head_ix = torch.index_select(edge_index[0], 0, mask_ix[1])
        all_tail_ix = torch.index_select(edge_index[1], 0, mask_ix[1])

        # Convert indices to offset based on the number of entities and batch num
        offset_head_ix = all_head_ix + (num_ents * mask_batch_num)
        offset_tail_ix = all_tail_ix + (num_ents * mask_batch_num)

        if self.dependent:
            relation = self.relation_linear(query).view(batch_size * self.num_relation, self.input_dim)

            offset_rel_ix = all_rel_ix + (self.num_relation * mask_batch_num)
            rel_embs = torch.index_select(relation, 0, offset_rel_ix)
        else:
            relation = self.relation.weight.expand(batch_size, -1, -1)
            relation = relation.reshape(-1, self.input_dim)
            rel_embs = self.relation(all_rel_ix)

        head_embs = torch.index_select(input, 0, offset_head_ix)
        msgs = head_embs * rel_embs

        if self.aggregate_func == "pna":
            degree_agg = torch.zeros((input.shape[0],)).to(input.device)            
            degree_agg[offset_tail_ix] = torch.index_select(degree_in, 0, all_tail_ix)
            degree_agg = degree_agg.unsqueeze(0).unsqueeze(-1)
        else:
            degree_agg = None

        out = self.agg_func(msgs, offset_tail_ix, num_ents, batch_size, degree_agg, boundary, ni_msgs)
        out = self.update(out, input)

        return out



    def aggregate_with_boundary(self, input, tail_index, num_ents, batch_size, degree_agg, boundary, ni_msgs):    
        """
        Include boundary condition
        """
        final_dim_size = batch_size * num_ents

        if self.aggregate_func == "pna":        
            agg_sum = scatter(input, tail_index, dim=0, dim_size=final_dim_size, reduce="sum")
            agg_max = scatter(input, tail_index, dim=0, dim_size=final_dim_size, reduce="max")
            agg_min = scatter(input, tail_index, dim=0, dim_size=final_dim_size, reduce="min")    
            sq_sum = scatter(input ** 2, tail_index, dim=0, dim_size=final_dim_size, reduce="sum")

            degree_agg = degree_agg + 1 # Avoid 0...

            if self.agg_degree_msgs:
                ni_msgs = ni_msgs.reshape(batch_size * num_ents, 1)
                degree_msgs = ni_msgs * self.degree_bias

                # Those msgs are zero @ layer 1
                if self.layer_num == 1:
                    degree_msgs = ni_msgs * self.degree_bias * 0
                    denominator = degree_agg.squeeze(0)
                else:
                    degree_msgs = ni_msgs * self.degree_bias
                    denominator = degree_agg.squeeze(0) + ni_msgs

                agg_max = torch.max(torch.max(agg_max, boundary), degree_msgs)
                agg_min = torch.max(torch.min(agg_min, boundary), degree_msgs)

                # With self-loop + Degree msgs
                agg_mean = (agg_sum + boundary + degree_msgs) / denominator
                agg_mean = agg_mean.squeeze(0)
                
                agg_std = (sq_sum + boundary ** 2 + degree_msgs ** 2).clamp(min=1e-6) / denominator
                agg_std = agg_std.squeeze(0)
            else:
                # With self-loop + Degree msgs
                agg_mean = (agg_sum + boundary) / degree_agg
                agg_mean = agg_mean.squeeze(0)
                
                agg_std = (sq_sum + boundary ** 2).clamp(min=1e-6) / degree_agg
                agg_std = agg_std.squeeze(0)

            features = [agg_mean.unsqueeze(-1), agg_max.unsqueeze(-1), agg_min.unsqueeze(-1), agg_std.unsqueeze(-1)]
            features = torch.cat(features, dim=-1)
            features = features.flatten(-2)

            scale = torch.log(degree_agg)
            scale = scale / scale.mean()

            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2).squeeze(0)
        else:
            output = scatter(input, tail_index, dim=0, dim_size=final_dim_size, reduce=self.aggregate_func)

        return output


    def aggregate_wo_boundary(self, input, tail_index, num_ents, batch_size, degree_agg, boundary, ni_msgs):    
        """
        Don't include boundary condition
        """
        final_dim_size = batch_size * num_ents

        if self.aggregate_func == "pna":
            agg_max = scatter(input, tail_index, dim=0, dim_size=final_dim_size, reduce="sum")
            agg_min = scatter(input, tail_index, dim=0, dim_size=final_dim_size, reduce="sum")

            # Add degree_i * degree_emb msgs to each node i
            if self.agg_degree_msgs:
                agg_mean = scatter(input, tail_index, dim=0, dim_size=final_dim_size, reduce="sum")
                sq_mean = scatter(input ** 2, tail_index, dim=0, dim_size=final_dim_size, reduce="sum")

                ni_msgs = ni_msgs.reshape(batch_size * num_ents, 1)
                degree_msgs = ni_msgs * self.degree_bias

                # Those msgs are zero @ layer 1
                if self.layer_num == 1:
                    degree_msgs = ni_msgs * self.degree_bias * 0
                    denominator = degree_agg.squeeze(0)
                else:
                    degree_msgs = ni_msgs * self.degree_bias
                    denominator = degree_agg.squeeze(0) + ni_msgs

                agg_max = torch.max(agg_max, degree_msgs)
                agg_min = torch.min(agg_min, degree_msgs)

                agg_mean = (agg_mean + degree_msgs) / (denominator + 1)   # NOTE: agg_mean in righthand expression is really sum
                agg_std = (sq_mean + degree_msgs ** 2).clamp(min=1e-6).sqrt() / (denominator + 1)  # NOTE: Same as with agg_mean
            else:
                agg_mean = scatter(input, tail_index, dim=0, dim_size=final_dim_size, reduce="mean")
                sq_mean = scatter(input ** 2, tail_index, dim=0, dim_size=final_dim_size, reduce="mean")
                agg_std = (sq_mean - agg_mean ** 2).clamp(min=1e-6).sqrt()

            features = torch.cat([agg_mean, agg_max, agg_min, agg_std], dim=-1)

            degree_agg = degree_agg.squeeze(0)
            scale = torch.log(degree_agg + 1e-6)
            scale = scale / scale.mean()

            scale_feats = [features, features * scale, features * (1 / scale.clamp(min=1e-2))]
            output = torch.cat(scale_feats, dim=-1)
        else:
            output = scatter(input, tail_index, dim=0, dim_size=final_dim_size, reduce=self.aggregate_func)
                    

        return output


    def update(self, update, input):
        """
        """  
        output = self.linear(torch.cat([input, update], dim=-1))   
        output = self.layer_norm(output)

        if self.activation:
            output = self.activation(output)

        if self.drop and self.training:
            output = F.dropout(output, p=self.drop)
        
        return output

