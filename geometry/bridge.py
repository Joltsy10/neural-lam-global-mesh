import numpy as np
import torch
from pathlib import Path

from geometry.icosahedron import get_icosahedron
from geometry.subdivison import refine
from geometry.cartesion import latlon_to_cartesian, cartesian_to_latlon
from geometry.g2m import build_g2m_edges, angular_to_euclidean_radius
from geometry.m2g import build_m2g
from geometry.hierarchy import build_hierarchy, get_inter_level_edges