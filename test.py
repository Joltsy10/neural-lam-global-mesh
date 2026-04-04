import torch
import torch.nn as nn

# Old implementation
class OldMessagePassingLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, src_features, dst_features, edge_index,
                edge_features, n_dst_nodes=None):
        if n_dst_nodes is None:
            n_dst_nodes = dst_features.shape[0]
        src = edge_index[0]
        dst = edge_index[1]
        msg_input = torch.cat([src_features[src], edge_features], dim=-1)
        messages = self.message_mlp(msg_input)
        aggregated = torch.zeros(n_dst_nodes, messages.shape[1])
        aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        update_input = torch.cat([dst_features, aggregated], dim=-1)
        new_dst = self.update_mlp(update_input)
        return self.norm(dst_features + new_dst)

from  model.message_passing import MessagePassingLayer

torch.manual_seed(42)
nodes = torch.randn(100, 7)
edge_index = torch.randint(0, 100, (2, 400))
edges = torch.randn(400, 3)

old = OldMessagePassingLayer(node_dim=7, edge_dim=3, hidden_dim=64)
new = MessagePassingLayer(node_dim=7, edge_dim=3, hidden_dim=64)

# copy weights so both layers are identical
new.load_state_dict(old.state_dict())

out_old = old(nodes, nodes, edge_index, edges)
out_new = new(nodes, nodes, edge_index, edges)

print("Max diff:", (out_old - out_new).abs().max().item())
print("Match:", torch.allclose(out_old, out_new, atol=1e-5))