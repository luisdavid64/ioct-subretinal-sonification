import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from torch import classes
from transmitter import OSC_Transmitt_Process
from settings_sonify import *
from collections import deque


transmitter = OSC_Transmitt_Process(ip="127.0.0.1", port=12001)

zeta_target = 0.6  # try 0.5–0.7


def sonify(intensity_range, averaged_values, force):
    ext_averaged = np.repeat(np.array(averaged_values), mass_per_bucket)

    profile = []
    for i in range(len(ext_averaged)):
        line = (0, i, 0, ext_averaged[i])
        profile.append(line)

    indexes = [(t[0], t[1], t[2]) for t in profile]
    springList = get_edge_list(indexes, 1)

    for line in profile:
        label = line[1]//mass_per_bucket

        map_oscMass = interp1d(intensity_range, target_ranges[int(label)][0], kind='slinear', fill_value='extrapolate')
        mass = map_oscMass(line[3])
        transmitter.oscTransmittProc_OSC_Param(line, oscMass=mass)

    for spdmp in springList:
        node1, node2 = profile[spdmp[0]], profile[spdmp[1]]
        label = node1[1]//mass_per_bucket

        map_sprStf = interp1d(intensity_range, target_ranges[int(label)][1], kind='slinear', fill_value='extrapolate')
        map_sprDmp = interp1d(intensity_range, target_ranges[int(label)][2], kind='slinear', fill_value='extrapolate')

        f_sprStf_dec3 = np.round(map_sprStf(averaged_values[int(label)]), 3)
        f_sprDmp_dec4 = np.round(map_sprDmp(averaged_values[int(label)]), 4)

        transmitter.oscTransmittProc_SprD_Param(node1, node2, sprStf=f_sprStf_dec3, sprDmp=f_sprDmp_dec4)

    transmitter.oscTransmittProc_Toggle_with_force(force)

def sonify_with_force_and_pos(force, target_node, num_nodes_x=16):
    driver_node = f"m_{target_node[1]}_{target_node[0]}_0"
    listener_node = f"m_{num_nodes_x-1}_{target_node[0]}_0"
    print(f"Target Node: {target_node}, Listener Node: {listener_node}, Force: {force}")
    transmitter.oscTransmittProc_Toggle_with_force_and_driver_idx(force, driver_node, listener_node)

def sonify_ascan(forces, separate_f_components=False, ramp=False):
    if separate_f_components:
        for i in range(len(forces[0])):
            transmitter.ocsTransmittProc_MultiToggleSep(list(forces[:, i, :].flatten()), ramp=ramp)
    else:
        for i in range(len(forces[0])):
            transmitter.ocsTransmittProc_MultiToggle(list(forces[:, i]), ramp=ramp)

def sonify_ILM_RPE(f_ilm, f_rpe):
    transmitter.ocsTransmittProc_ToggleILM_RPE(f_ilm, f_rpe)

def get_edge_list(indexes, min_dist):

    distance = cdist(indexes, indexes, "euclidean")
    edge_nodes = np.where(distance <= min_dist)
    edge_nodes = np.array(edge_nodes).transpose()
    edge_nodes = edge_nodes[edge_nodes[:, 0] < edge_nodes[:, 1]]

    return edge_nodes


def exp_interplt(x, interp_func):
    log_x = np.log(x)
    log_y = interp_func(log_x)
    y = np.exp(log_y)

    return y

OFFSETS = [[ 1, 0 ], [ 0, 1 ], [1,1]]

def compute_plate_ids(classes):
    H, W = classes.shape
    plate_id = -np.ones_like(classes, dtype=int)
    current_id = 0
    
    for i in range(H):
        for j in range(W):
            if plate_id[i,j] != -1:
                continue  # already visited
                
            # BFS flood fill for this component
            q = deque([(i,j)])
            plate_id[i,j] = current_id
            class_value = classes[i,j]
            
            while q:
                y,x = q.popleft()
                
                for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                    ny,nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if plate_id[ny,nx] == -1 and classes[ny,nx] == class_value:
                            plate_id[ny,nx] = current_id
                            q.append((ny,nx))
            
            current_id += 1

    return plate_id

def fix_plate_corners(classes):
    H, W = classes.shape

    for c in np.unique(classes):
        ys, xs = np.where(classes == c)

        if len(xs) == 0:
            continue

        top    = ys.min()
        bottom = ys.max()
        left   = xs.min()
        right  = xs.max()

        # Fix the four corners
        corners = [
            (top, left),
            (top, right),
            (bottom, left),
            (bottom, right)
        ]

        for (i, j) in corners:
            print(f"Fixing corner node {(i, j, 0)} for class {c}")
            send_fixed_mass_node((i, j, 0))

def compute_springs_to_remove(plate_id, offsets):
    H, W = plate_id.shape
    springs_to_remove = []

    for i in range(H):
        for j in range(W):
            for dy,dx in offsets:
                ni, nj = i+dy, j+dx
                if 0 <= ni < H and 0 <= nj < W:
                    if plate_id[i,j] != plate_id[ni,nj]:
                        springs_to_remove.append(((i,j,0),(ni,nj,0)))
    
    return springs_to_remove

def compute_fixed_masses(plate_id):
    fixed = set()
    H, W = plate_id.shape
    plates = np.unique(plate_id)

    for p in plates:
        ys, xs = np.where(plate_id == p)
        top = ys.min()
        bottom = ys.max()
        left = xs.min()
        right = xs.max()

        # Four corners of bounding box for this plate
        corners = [
            (top, left),
            (top, right),
            (bottom, left),
            (bottom, right)
        ]
        for c in corners:
            fixed.add((c[0], c[1], 0))

    return list(fixed)

node_id_cache = None

def set_sonification_params(
        config,
        means,
        std,
        damping,
        classes,
        remove_diags=True,
        target_classes=[2,3],
        weaken_links=True,
        separate_plates=False):

    H, W = means.shape

    # ------------ PRECOMPUTE CONSTANTS ------------
    global node_id_cache
    if node_id_cache is None:
        node_id_cache = {(i, j): (i, j, 0) for i in range(H) for j in range(W)}

    # ------------ SEPARATE PLATES ------------
    if separate_plates:
        plate_id = compute_plate_ids(classes)
        springs_to_remove = compute_springs_to_remove(plate_id, OFFSETS)
        masses_to_fix = compute_fixed_masses(plate_id)

        for a, b in springs_to_remove:
            transmitter.oscTransmittProc_remove_spring(a, b)
        
        for m in masses_to_fix:
            send_fixed_mass_node(m)

    # ------------ SET MASSES (vectorized loop) ------------
    for (i, j), nid in node_id_cache.items():
        # if i == 1 or i == H-2 or j == 1 or j == W-2:
        #     transmitter.oscTransmittProc_OSC_Param(
        #         nid,
        #         oscMass=means[i,j] * 2 + 1e-2  # very
        #     )
        if i == 0 or i == H-1 or j == 0 or j == W-1:
            transmitter.oscTransmittProc_OSC_Param(
                nid,
                oscMass=means[i,j] * 4 + 1e-2  # very
            )
        else:
            transmitter.oscTransmittProc_OSC_Param(
                nid,
                oscMass=means[i, j] + 1e-2
            )

    # ------------ SET SPRINGS (optimized inner loops) ------------
    for i in range(H):
        for j in range(W):

            cid = classes[i, j]
            nid1 = node_id_cache[(i, j)]  # reuse tuple

            for di, dj in OFFSETS:
                ni = i + di
                nj = j + dj

                if ni >= H or nj >= W:
                    continue

                nid2 = node_id_cache[(ni, nj)]
                cid2 = classes[ni, nj]

                # --- Base stiffness/damping ---
                stf = (std[i,j] * std[ni,nj]) ** 0.5
                dmp = 0.5 * (damping[i,j] + damping[ni,nj])

                # --- If crossing layers, use stronger damping ---
                if cid != cid2:
                    dmp = max(damping[i, j], damping[ni, nj])

                    if weaken_links:
                        stf *= 0.5
                        dmp *= 1.2

                # --- Remove diagonal springs except for target layer ---
                if remove_diags and di == 1 and dj == 1 and cid not in target_classes:
                    stf = 0
                    dmp = 0

                # --- Send spring param ---
                transmitter.oscTransmittProc_SprD_Param(
                    nid1, nid2, sprStf=stf, sprDmp=dmp
                )


def send_debug_message(message):
    transmitter.oscTransmittProc_Debug_message(message)

def start_recording():
    print("Starting recording...")
    transmitter.oscTransmittProc_RecordStart()

def stop_recording():
    print("Stopping recording...")
    transmitter.oscTransmittProc_RecordEnd()

def send_rescale_params(stiff, damp):
    # print(f"Rescaling stiffness by factor: {stiff}, damping by factor: {damp}")
    transmitter.oscTransmittProc_RescaleParams(stiff, damp)

def send_fixed_mass_node(line):
    print(f"Fixing mass node: {line}")
    transmitter.oscTransmittProc_FixNode(line)

def move_listener_node(line):
    print(f"Moving listener node to: {line}")
    transmitter.oscTransmittProc_MoveListener(line)

def set_crackle_params(density, gain):
    density = max(0.0, min(density, 10.0))
    gain = max(0.0, min(gain, 5.0))
    # print(f"Setting crackle params - Density: {density}, Gain: {gain}")
    transmitter.oscTransmittProc_SetCrackle(density, gain)

def set_amfm_params(def_value):
    # print(f"Setting AM/FM params - Def: {def_value}")
    transmitter.oscTransmittProc_SetAMFM(def_value)

def reset_amfm_params():
    transmitter.oscTransmittProc_ResetAMFM()

def set_effective_stiffening_params(stiffness_matrix, damping_matrix=None, classes=None, 
                                   remove_diags=True, target_classes=[2,3], weaken_links=False):
    """
    Updates only the stiffness parameters for springs in the physical model.
    Maps rest distance changes (tissue deformation) to updated stiffness values.
    Uses the same logic as set_sonification_params for consistency.
    
    Args:
        stiffness_matrix (np.ndarray): 2D array of stiffness values with shape (H, W)
                                      where H and W match the grid dimensions
        damping_matrix (np.ndarray, optional): 2D array of damping values. 
                                              If None, uses default damping of 0.0001
        classes (np.ndarray, optional): 2D array of class labels for advanced logic
        remove_diags (bool): Whether to remove diagonal springs except for target class
        target_class (int): Class that keeps diagonal springs when remove_diags=True
        weaken_links (bool): Whether to weaken springs crossing between different classes
    """
    H, W = stiffness_matrix.shape
    
    # Use cached node IDs if available
    global node_id_cache
    if node_id_cache is None:
        node_id_cache = {(i, j): (i, j, 0) for i in range(H) for j in range(W)}
    
    # Set spring stiffness for all connections
    for i in range(H):
        for j in range(W):
            nid1 = node_id_cache[(i, j)]
            
            # Get class info if available
            cid = classes[i, j] if classes is not None else 0
            
            for di, dj in OFFSETS:  # [(1, 0), (0, 1), (1, 1)]
                ni = i + di
                nj = j + dj
                
                if ni >= H or nj >= W:
                    continue
                
                nid2 = node_id_cache[(ni, nj)]
                cid2 = classes[ni, nj] if classes is not None else 0
                
                # --- Base stiffness/damping (same logic as set_sonification_params) ---
                stiffness = min(stiffness_matrix[i, j], stiffness_matrix[ni, nj])
                
                if damping_matrix is not None:
                    damping = min(damping_matrix[i, j], damping_matrix[ni, nj])
                else:
                    damping = 0.0001  # Default damping value
                
                # --- If crossing layers, use stronger damping ---
                if classes is not None and cid != cid2:
                    if damping_matrix is not None:
                        damping = max(damping_matrix[i, j], damping_matrix[ni, nj])
                    
                    if weaken_links:
                        stiffness *= 0.5
                        damping *= 1.2
                
                # --- Remove diagonal springs except for target layer ---
                if remove_diags and di == 1 and dj == 1 and cid not in target_classes:
                    stiffness = 0
                    damping = 0

                # Send spring parameter update
                transmitter.oscTransmittProc_SprD_Param(
                    nid1, nid2, 
                    sprStf=float(stiffness), 
                    sprDmp=float(damping)
                )
      
