import numpy as np

def analyze_trajectory(traj_data, env=None, verbose=False):
    """
    Analyzes trajectory data structure and statistics.
    Designed for use during debugging (pdb/gdb) or runtime inspection.
    
    Args:
        traj_data (dict): The trajectory data dictionary loaded from LoadTranData.
        env (object, optional): The environment instance to retrieve configuration (save_freq, intervals).
        verbose (bool): If True, prints detailed recursive structure analysis.
    """
    print("\n" + "="*40)
    print(" [Trajectory Analysis Tool] ")
    print("="*40)

    # --- 1. Structure Overview ---
    print(f"Type: {type(traj_data)}")
    if isinstance(traj_data, dict):
        print(f"Keys: {list(traj_data.keys())}")
        
    if verbose:
        print("\n[Structure Details]")
        _print_structure_recursive(traj_data)

    # --- 2. Statistical Analysis ---
    print("\n[Statistics]")
    
    # Identify primary path key for length calculation
    # Common keys in this project: 'left_joint_path', 'qpos', 'actions'
    path_key = 'left_joint_path'
    if path_key not in traj_data:
        # Fallback: look for any list
        for k, v in traj_data.items():
            if isinstance(v, list):
                path_key = k
                break
    
    if path_key in traj_data:
        data_seq = traj_data[path_key]
        if isinstance(data_seq, list):
            # Check if segmented (list of lists) or flat
            is_segmented = False
            total_len = 0
            
            if len(data_seq) > 0:
                first_elem = data_seq[0]
                if isinstance(first_elem, (list, np.ndarray)) or (hasattr(first_elem, '__len__') and not isinstance(first_elem, (str, dict))):
                    # Likely segmented
                    is_segmented = True
                    total_len = sum(len(seg) for seg in data_seq)
                else:
                    total_len = len(data_seq)
            
            print(f"Target Key: '{path_key}'")
            print(f"Structure: {'Segmented List' if is_segmented else 'Flat List'}")
            print(f"Total Steps (Sum): {total_len}")
            
            if env:
                # Retrieve Env Configs
                save_freq = getattr(env, 'save_freq', None)
                if save_freq is None: 
                    save_freq = 1
                    
                phase_intervals = getattr(env, 'phase_intervals', 'Not Set')
                
                effective_len = total_len // save_freq if save_freq > 0 else 0
                
                print(f"Save Frequency (env.save_freq): {save_freq}")
                print(f"Effective Record Length: {effective_len} frames")
                print(f"Sampling Intervals (env.phase_intervals): {phase_intervals}")
    else:
        print("No list-like data found to analyze length.")

    print("="*40 + "\n")

def _print_structure_recursive(data, indent=0, max_depth=3, depth=0):
    if depth > max_depth:
        return
    
    prefix = "  " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            _print_info(k, v, prefix, indent, max_depth, depth)
    elif isinstance(data, list):
        print(f"{prefix}List (len={len(data)})")
        if len(data) > 0:
            print(f"{prefix}  Sample Element[0]:")
            _print_structure_recursive(data[0], indent + 2, max_depth, depth + 1)

def _print_info(key, val, prefix, indent, max_depth, depth):
    if isinstance(val, list):
        print(f"{prefix}- {key}: List (len={len(val)})")
        if len(val) > 0 and depth < max_depth:
             traj_elem = val[0]
             print(f"{prefix}  Type of Elem[0]: {type(traj_elem)}")
             if hasattr(traj_elem, 'shape'):
                  print(f"{prefix}  Shape of Elem[0]: {traj_elem.shape}")
             elif isinstance(traj_elem, dict):
                  # Recurse briefly
                  _print_structure_recursive(traj_elem, indent + 2, max_depth, depth + 1)
    elif hasattr(val, 'shape'):
        print(f"{prefix}- {key}: Array shape={val.shape}")
    else:
        print(f"{prefix}- {key}: {type(val)}")
