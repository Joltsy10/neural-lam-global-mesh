import torch

m2m_ei   = torch.load("graph_output/m2m_edge_index.pt")
mesh_f   = torch.load("graph_output/mesh_features.pt")
g2m_ei   = torch.load("graph_output/g2m_edge_index.pt")
m2g_ei   = torch.load("graph_output/m2g_edge_index.pt")
up_ei    = torch.load("graph_output/mesh_up_edge_index.pt")
down_ei  = torch.load("graph_output/mesh_down_edge_index.pt")
up_f     = torch.load("graph_output/mesh_up_features.pt")
down_f   = torch.load("graph_output/mesh_down_features.pt")
m2m_f    = torch.load("graph_output/m2m_features.pt")

print(f"M2M levels: {len(m2m_ei)}")
for i, ei in enumerate(m2m_ei):
    print(f"  level {i}: {ei.shape}")

print(f"Mesh features levels: {len(mesh_f)}")
for i, f in enumerate(mesh_f):
    print(f"  level {i}: {f.shape}")

print(f"Up edge levels: {len(up_ei)}")
for i, ei in enumerate(up_ei):
    print(f"  level {i}: {ei.shape}")

print(f"G2M: {g2m_ei.shape}")
print(f"M2G: {m2g_ei.shape}")

print(f"Hierarchical detected (len > 1): {len(m2m_ei) > 1}")
print(f"Any nan in m2m features: {any(f.isnan().any() for f in m2m_f)}")
print(f"Any nan in up features:  {any(f.isnan().any() for f in up_f)}")
print(f"Any nan in down features:{any(f.isnan().any() for f in down_f)}")