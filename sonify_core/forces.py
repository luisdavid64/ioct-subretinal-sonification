import numpy as np
from utils.util import compute_needle_force_proxy
def compute_magnitudes(force_info, needle_speed, vx, vy, sep_f_components, use_handle_forces):
    """Return normalized magnitudes (NxM or 1D) depending on mode."""
    if force_info is not None:
        key = "f_comps" if sep_f_components else "magnitudes"
        magnitudes = 100 * np.array(force_info[key])
        if use_handle_forces and "handle_forces" in force_info:
            handle_forces = np.array(force_info["handle_forces"]) / 1000
            magnitudes += handle_forces.reshape(-1, 1)
    else:
        proxy = compute_needle_force_proxy(needle_speed, vx, vy)
        if sep_f_components:
            magnitudes = proxy["f_comps"] / 100
        else:
            magnitudes = np.array([proxy["tip_force_norm"] / 100])
    return magnitudes

def compute_node_magnitudes(num_nodes_y, magnitudes, closest_node_idx, dist,
                            deflection_scaling, sigma=5, norm="l1"):
    """Distribute magnitudes across nodes with Gaussian weighting, forward direction only."""

    magnitudes = np.asarray(magnitudes)
    y_indices = np.arange(num_nodes_y)
    dy = y_indices - closest_node_idx[0]  # positive means forward/downward

    # Forward-only mask
    forward_mask = dy >= 0

    # Apply Gaussian only to forward nodes
    influence = np.exp(- (dy ** 2) / (2 * sigma ** 2))
    influence[~forward_mask] = 0.0  # zero out backward influence

    # Normalize so all influence sums to 1
    if influence.sum() > 0:
        influence /= influence.sum()

    influence = influence.reshape((num_nodes_y,) + (1,) * magnitudes.ndim)

    node_magnitudes = influence * magnitudes * deflection_scaling

    return node_magnitudes

def compute_node_magnitudes_forward(
    num_nodes_y,
    magnitudes,
    closest_node_idx,
    dist,
    deflection_scaling,
    sigma=5,
):
    """
    Distribute magnitudes across nodes ONLY in the forward (downward) direction
    of the needle movement. Physiologically correct for retinal indentation.
    """

    magnitudes = np.asarray(magnitudes)
    y_indices = np.arange(num_nodes_y)
    dy = y_indices - closest_node_idx[0]  # positive means forward/downward

    # Forward-only mask
    forward_mask = dy >= 0

    # Apply Gaussian only to forward nodes
    influence = np.exp(- (dy ** 2) / (2 * sigma ** 2))
    influence[~forward_mask] = 0.0  # zero out backward influence

    # Normalize so all influence sums to 1
    if influence.sum() > 0:
        influence /= influence.sum()

    # Broadcast influence to match magnitude dimensions
    influence = influence.reshape((num_nodes_y,) + (1,) * magnitudes.ndim)

    node_magnitudes = influence * magnitudes * deflection_scaling

    return node_magnitudes