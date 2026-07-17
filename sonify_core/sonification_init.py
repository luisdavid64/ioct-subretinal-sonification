import json

import numpy as np
import torch

TEMPLATE_PATH = 'sonification_configs/template.json'

def compute_listeners(classes_per_patch, target_classes=[2,3]):
    H, W = classes_per_patch.shape
    listeners = []

    # Find horizontal interfaces (transitions between classes in rows)
    for row in range(H - 1):
        for col in [1,W-2]:
            current_class = classes_per_patch[row, col]
            next_class = classes_per_patch[row + 1, col]
            
            # Check if there's a transition between target classes
            if (next_class in target_classes and current_class != next_class):
                # Place listener at the interface (between current and next column)
                interface_col = col
                listeners.append(f"m_{interface_col}_{row}_0")
    
    listeners.append(f"m_{W-2}_{H-1}_0")
    listeners.append(f"m_{1}_{H-1}_0")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_listeners = []
    for listener in listeners:
        if listener not in seen:
            seen.add(listener)
            unique_listeners.append(listener)
    
    return unique_listeners

def sound_model_config_init(ROI, frame_roi, seg_roi, num_nodes_x=16, num_nodes_y=16, classes=None):
    """ 
        Initialize sound model configuration
        based on ROI and image patch. 
        
        Returns config and processed means/stds matrices for grid geometry
    """
    classes_per_patch = classes.reshape((num_nodes_y, num_nodes_x))
    config = json.load(open(TEMPLATE_PATH, 'r'))
    config["model"] = "3D"
    config['geometry']['numLayers'] = 1 
    config['geometry']['dy'] = num_nodes_y
    config['geometry']['dx'] = num_nodes_x
    config["geometry"]["numNodesPerLayer"] = [num_nodes_y]
    config['geometry']['dz'] = 1
    config["sonification_set_up"]["drivers"] = [f"m_{num_nodes_x // 2}_{i}_0" for i in range(num_nodes_y)]
    # Assign one listener every 3 nodes in a column
    config["sonification_set_up"]["listeners"] = list(reversed(compute_listeners(classes_per_patch)))
    config["geometry"]["distance"] = 3.2
    config["global_friction"] = 0.08
    
    # frame_roi to grayscale
    frame_roi = (frame_roi - frame_roi.min()) / (frame_roi.max() - frame_roi.min() + 1e-6)
    if len(frame_roi.shape) == 3:
        # rescale from 0 to 1
        frame_roi = np.mean(frame_roi, axis=-1)

    ROI_patches = np.reshape(frame_roi, (num_nodes_y, num_nodes_x, -1))
    seg_patches = np.reshape(seg_roi,   (num_nodes_y, num_nodes_x, -1))

    means = ROI_patches.mean(axis=(2))
    stds = ROI_patches.std(axis=(2)) + 1e-6  # avoid division by zero

    save_config(config, '/tmp/son_config.json')

    print("Initialized sonification config and geometry.")
    print(f"Path: /tmp/son_config.json")
    print(f"Source: /tmp/son_source.npy")

    return config, means, stds

def add_ILM_RPE_drivers(ilm_bound, rpe_bound, num_nodes_x, num_nodes_y, include_retina_in_ilm_rpe_drivers=False):
    config = json.load(open('/tmp/son_config.json', 'r'))
    config["sonification_set_up"]["drivers_ilm"] = [f"m_{0}_{ilm_bound}_{0}", f"m_{num_nodes_x-1}_{ilm_bound}_{0}"]
    if include_retina_in_ilm_rpe_drivers:
        for i in range(ilm_bound, rpe_bound):
            config["sonification_set_up"]["drivers_ilm"].append(f"m_{0}_{i}_{0}")
            config["sonification_set_up"]["drivers_ilm"].append(f"m_{num_nodes_x-1}_{i}_{0}")
    config["sonification_set_up"]["drivers_rpe"] = [f"m_{0}_{rpe_bound}_{0}", f"m_{num_nodes_x-1}_{rpe_bound}_{0}"]
    save_config(config, '/tmp/son_config.json')


def create_sonification_geometry(ROI, frame_roi, seg_roi, num_nodes_x=16, num_nodes_y=16):
    """
    Complete pipeline: create config and geometry from image data
    
    Args:
        ROI: Region of interest
        frame_roi: Frame region of interest
        seg_roi: Segmentation region of interest  
        num_nodes_x: Number of nodes in x direction
        num_nodes_y: Number of nodes in y direction
    
    Returns:
        config: Configuration dictionary
        nodes: Node tensor for physics simulation
        edges: Edge tensor for spring connections
        springs: Spring parameters tensor
        means: Processed means matrix
        stds: Processed stds matrix
    """
    # Get config and processed matrices
    config, means, stds = sound_model_config_init(ROI, frame_roi, seg_roi, num_nodes_x, num_nodes_y)
    
    # Build geometry from the processed matrices
    nodes, edges, springs = build_geometry_from_grid(means, stds, config)
    
    return config, nodes, edges, springs, means, stds


def build_geometry_from_grid(means, stds, config):
    """
    Build geometry from regular 2D grid using means as masses and stds as stiffnesses
    
    Args:
        means: 2D array (num_nodes_y, num_nodes_x) - will be used as masses
        stds: 2D array (num_nodes_y, num_nodes_x) - will be used as stiffnesses
        config: configuration dictionary
    
    Returns:
        nodes, edges, springs tensors for the grid geometry
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    num_nodes_y, num_nodes_x = means.shape
    
    # Create regular grid coordinates
    # Normalize to have reasonable spacing based on config distance
    spacing = config["geometry"]["distance"] # in mm
    
    # Create grid vertices
    x_coords = torch.linspace(0, (num_nodes_x - 1) * spacing, num_nodes_x, device=device)
    y_coords = torch.linspace(0, (num_nodes_y - 1) * spacing, num_nodes_y, device=device)
    
    # Create meshgrid and flatten to get all vertex positions
    X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    Z = torch.zeros_like(X)  # 2D grid in XY plane
    
    # Flatten and stack to get vertices (num_nodes, 3)
    vertices = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    num_nodes = vertices.shape[0]
    
    # Convert means and stds to tensors and flatten
    masses_grid = torch.tensor(means, dtype=dtype, device=device).flatten().unsqueeze(1)
    stiffness_grid = torch.tensor(stds, dtype=dtype, device=device).flatten()
    
    # Node properties
    radius = torch.full((num_nodes, 1), config["geometry"]["massesRadius"], device=device)
    fixed = torch.zeros((num_nodes, 1), device=device)
    drivers = torch.zeros((num_nodes, 1), device=device)
    listeners = torch.zeros((num_nodes, 1), device=device)
    
    # Create nodes tensor: [x, y, z, mass, radius, fixed, drivers, listeners]
    nodes = torch.cat([vertices, masses_grid, radius, fixed, drivers, listeners], dim=1)
    
    # --- Create edges for regular grid connectivity
    edges_list = []
    springs_list = []
    
    # Helper function to convert 2D grid indices to flat index
    # means[y][x] -> flat index = y * num_nodes_x + x
    def grid_to_flat(x, y):
        return y * num_nodes_x + x
    
    # Create horizontal and vertical connections
    for x in range(num_nodes_x):
        for y in range(num_nodes_y):
            current_idx = grid_to_flat(x, y)
            
            # Horizontal connection (right neighbor)
            if x < num_nodes_x - 1:
                neighbor_idx = grid_to_flat(x + 1, y)
                edges_list.append([current_idx, neighbor_idx])
                
                # Rest length is the spacing
                rest_len = spacing
                
                # Average stiffness of connected nodes
                k_avg = (stiffness_grid[current_idx] + stiffness_grid[neighbor_idx]) / 2.0
                
                # Use base damping from config
                z_base = torch.tensor(config["parameters"]["C"][0], device=device)
                
                springs_list.append([k_avg, z_base, rest_len, 0])  # 0 = horizontal connection
            
            # Vertical connection (down neighbor)
            if y < num_nodes_y - 1:
                neighbor_idx = grid_to_flat(x, y + 1)
                edges_list.append([current_idx, neighbor_idx])
                
                # Rest length is the spacing
                rest_len = spacing
                
                # Average stiffness of connected nodes
                k_avg = (stiffness_grid[current_idx] + stiffness_grid[neighbor_idx]) / 2.0
                
                # Use base damping from config
                z_base = torch.tensor(config["parameters"]["C"][0], device=device)
                
                springs_list.append([k_avg, z_base, rest_len, 1])  # 1 = vertical connection
    
    # Convert to tensors
    edges = torch.tensor(edges_list, dtype=torch.long, device=device).T
    springs = torch.tensor(springs_list, dtype=dtype, device=device)
    
    print(f"Created grid geometry: {num_nodes_x}x{num_nodes_y} = {num_nodes} nodes, {len(edges_list)} springs")
    print(f"Mass range: {masses_grid.min().item():.4f} - {masses_grid.max().item():.4f}")
    print(f"Stiffness range: {stiffness_grid.min().item():.4f} - {stiffness_grid.max().item():.4f}")
    
    return nodes, edges, springs


def build_geometry_from_mesh(mesh_path, config):
    mesh = trimesh.load(mesh_path, process=True)
    # assert mesh.is_watertight or mesh.is_volume, "Non-watertight mesh might behave oddly (still usable though)."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # --- Vertices → nodes
    vertices = torch.tensor(mesh.vertices, dtype=dtype, device=device)
    num_nodes = vertices.shape[0]

    masses = torch.full((num_nodes, 1), config["parameters"]["M"][0], device=device)
    radius = torch.full((num_nodes, 1), config["geometry"]["massesRadius"], device=device)
    fixed = torch.zeros((num_nodes, 1), device=device)
    drivers = torch.zeros((num_nodes, 1), device=device)
    listeners = torch.zeros((num_nodes, 1), device=device)

    # you can later assign drivers/listeners by name or region
    nodes = torch.cat([vertices, masses, radius, fixed, drivers, listeners], dim=1)

    # --- Edges → springs
    # trimesh stores unique edges for faces
    edges = torch.tensor(mesh.edges_unique, dtype=torch.long, device=device).T
    i, j = edges
    rest_len = torch.norm(vertices[j] - vertices[i], dim=1)

    # stiffness & damping based on config
    k_base = torch.tensor(config["parameters"]["K"][0], device=device)
    z_base = torch.tensor(config["parameters"]["C"][0], device=device)

    springs = torch.stack([
        k_base.repeat(len(rest_len)),
        z_base.repeat(len(rest_len)),
        rest_len,
        torch.zeros_like(rest_len)  # neighbor_type (0=first)
    ], dim=1)

    return nodes, edges, springs

def save_geometry(nodes, edge_index, springs, path):
    np.savez(path, nodes=nodes.cpu().numpy(), edge_index=edge_index.cpu().numpy(), springs=springs.cpu().numpy())

def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def visualize_geometry(nodes, edges, springs):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot nodes
    ax.scatter(nodes[:, 0].cpu(), nodes[:, 1].cpu(), nodes[:, 2].cpu(), c='b', s=20)

    # Plot edges
    for edge in edges.T:
        p1 = nodes[edge[0], :3].cpu().numpy()
        p2 = nodes[edge[1], :3].cpu().numpy()
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='r', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title('Geometry Visualization')
    plt.show()
