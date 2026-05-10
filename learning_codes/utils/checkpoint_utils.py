import os
import torch
import pickle
from typing import Dict, Optional
from Region import RegionController 
import json
import shutil
from pathlib import Path
import re

def load_checkpoint(
    region_controllers: Dict[int, RegionController],
    path: str = "latest_checkpoint",
    map_location: Optional[torch.device | str] = None,
) -> Optional[dict]:
    """
    Loads the latest checkpoint into the provided region controllers, if present.

    Returns:
        dict with {"saved_episode": int, "next_episode": int} on success,
        or None if no checkpoint is found.
    """
    if not os.path.isdir(path):
        print(f"[load_checkpoint] No checkpoint dir at '{path}'. Fresh start.")
        return None

    # Load global episode metadata
    metadata_path = os.path.join(path, "metadata.json")
    if not os.path.isfile(metadata_path):
        print(f"[load_checkpoint] No metadata.json in '{path}'. Fresh start.")
        return None

    with open(metadata_path, "r") as f:
        meta = json.load(f)
    saved_ep = int(meta.get("episode", 0))
    next_ep = saved_ep + 1  # you saved after finishing ep, so continue with the next

    # Choose default map_location when not provided
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n--- Loading checkpoint from '{path}' (saved episode {saved_ep}) ---")
    for region_id, controller in region_controllers.items():
        # 1) torch state
        ckpt_pth = os.path.join(path, f"region_{region_id}_checkpoint.pth")
        if not os.path.isfile(ckpt_pth):
            print(f"[load_checkpoint] Missing torch state for region {region_id}: {ckpt_pth}")
            continue

        torch_state = torch.load(ckpt_pth, map_location=map_location)

        controller.policy_gnn.load_state_dict(torch_state["policy_gnn_state_dict"])
        controller.target_gnn.load_state_dict(torch_state["target_gnn_state_dict"])
        controller.policy_net.load_state_dict(torch_state["policy_net_state_dict"])
        controller.target_net.load_state_dict(torch_state["target_net_state_dict"])
        controller.optimizer.load_state_dict(torch_state["optimizer_state_dict"])

        # Scalars / hyperparams (keep your current code in sync with keys you save)
        controller.epsilon       = torch_state.get("epsilon", controller.epsilon)
        controller.update_step   = torch_state.get("update_step", controller.update_step)
        controller.gamma         = torch_state.get("gamma", controller.gamma)
        controller.epsilon_min   = torch_state.get("epsilon_min", controller.epsilon_min)
        controller.epsilon_decay = torch_state.get("epsilon_decay", controller.epsilon_decay)
        controller.learning_rate = torch_state.get("learning_rate", controller.learning_rate)
        controller.target_update = torch_state.get("target_update", controller.target_update)

        # 2) replay memory
        mem_pkl = os.path.join(path, f"region_{region_id}_memory.pkl")
        if os.path.isfile(mem_pkl):
            with open(mem_pkl, "rb") as f:
                controller.memory = pickle.load(f)
        else:
            print(f"[load_checkpoint] Missing memory for region {region_id}: {mem_pkl} (continuing without it)")

    print(f"--- Checkpoint loaded. Will resume from episode {next_ep}. ---\n")
    return {"saved_episode": saved_ep, "next_episode": next_ep}

def save_checkpoint(
    region_controllers: Dict[int, RegionController], 
    episode: int, 
    path: str = "latest_checkpoint"
):
    """
    Saves and overwrites a single checkpoint with the current training state.

    This includes:
    - Model state dictionaries (policy and target networks)
    - Optimizer state dictionary
    - Replay buffer (memory)
    - Current epsilon value
    - Training step counter (update_step)
    - The current episode number

    Args:
        region_controllers (Dict[int, RegionController]): Dictionary of all region controllers.
        episode (int): The current episode number to be stored in metadata.
        path (str): The single directory where the checkpoint will be overwritten.
    """
    # Create the single checkpoint directory if it doesn't exist.
    os.makedirs(path, exist_ok=True)

    print(f"\n--- Overwriting checkpoint for Episode {episode} in '{path}/' ---")

    # Save state for each region controller
    for region_id, controller in region_controllers.items():
        
        # 1. Define file paths for this region within the single directory
        checkpoint_filepath = os.path.join(path, f"region_{region_id}_checkpoint.pth")
        memory_filepath = os.path.join(path, f"region_{region_id}_memory.pkl")

        # 2. Prepare the state dictionary for PyTorch components
        torch_state = {
            'policy_gnn_state_dict': controller.policy_gnn.state_dict(),
            'target_gnn_state_dict': controller.target_gnn.state_dict(),
            'policy_net_state_dict': controller.policy_net.state_dict(),
            'target_net_state_dict': controller.target_net.state_dict(),
            'optimizer_state_dict': controller.optimizer.state_dict(),
            'epsilon': controller.epsilon,
            'update_step': controller.update_step,
            'gamma': controller.gamma,
            'epsilon_min': controller.epsilon_min,
            'epsilon_decay': controller.epsilon_decay,
            'learning_rate': controller.learning_rate,
            'target_update': controller.target_update
        }

        # 3. Save the PyTorch components, overwriting any previous file
        torch.save(torch_state, checkpoint_filepath)

        # 4. Save the replay buffer using pickle, overwriting any previous file
        with open(memory_filepath, 'wb') as f:
            pickle.dump(controller.memory, f)

    # 5. Save global metadata, overwriting the previous file
    metadata_filepath = os.path.join(path, "metadata.json")
    with open(metadata_filepath, 'w') as f:
        json.dump({'episode': episode}, f, indent=4)
        
    print(f"--- Checkpoint saved successfully. ---\n")


def _read_latest_simulation_metrics(log_path: str = "logs/simulation_metrics.txt") -> Dict[int, Dict[str, float]]:
    """
    Parse `logs/simulation_metrics.txt` created by `log_episode_metrics.py`.
    Returns a dict mapping episode -> {"avg_mtt": float or None, "avg_waiting": float or None}
    The file format expected (per episode block):
        Episode N:
          Average meanTravelTime : X
          Average waitingTime: Y

    If the file or values are missing, those entries will be None.
    """
    results = {}
    p = Path(log_path)
    if not p.exists():
        return results

    content = p.read_text(encoding="utf-8")
    # Split by 'Episode ' occurrences
    # We'll use regex to find Episode blocks
    pattern = re.compile(r"Episode\s+(\d+):\s*\n\s*Average meanTravelTime\s*:\s*(?P<mtt>[-0-9\.NoneNaN]+)\s*\n\s*Average waitingTime\s*:\s*(?P<wt>[-0-9\.NoneNaN]+)", re.MULTILINE)
    for m in pattern.finditer(content):
        ep = int(m.group(1))
        mtt_raw = m.group('mtt')
        wt_raw = m.group('wt')
        try:
            mtt = float(mtt_raw) if mtt_raw not in ("None", "nan", "NaN") else None
        except Exception:
            mtt = None
        try:
            wt = float(wt_raw) if wt_raw not in ("None", "nan", "NaN") else None
        except Exception:
            wt = None
        results[ep] = {"avg_mtt": mtt, "avg_waiting": wt}
    return results


def _ensure_top_dirs(base_path: str = "latest_checkpoint"):
    """Ensure TT/WT and 1..5 subfolders exist."""
    base = Path(base_path)
    tt = base / "TT"
    wt = base / "WT"
    for parent in (tt, wt):
        parent.mkdir(parents=True, exist_ok=True)
        for i in range(1, 6):
            (parent / str(i)).mkdir(parents=True, exist_ok=True)


def _load_rankings(base_path: str = "latest_checkpoint") -> Dict[str, list]:
    """Load or initialize rankings JSON file which stores list of dicts for TT and WT.
    Each list contains up to 5 entries: {"episode": int, "value": float}
    """
    base = Path(base_path)
    rank_file = base / "top5_rankings.json"
    if not rank_file.exists():
        return {"TT": [], "WT": []}
    try:
        return json.loads(rank_file.read_text(encoding="utf-8"))
    except Exception:
        return {"TT": [], "WT": []}


def _save_rankings(rankings: Dict[str, list], base_path: str = "latest_checkpoint"):
    base = Path(base_path)
    rank_file = base / "top5_rankings.json"
    rank_file.write_text(json.dumps(rankings, indent=2), encoding="utf-8")

def update_top5_checkpoints(region_controllers: Dict[int, RegionController], episode: int, base_path: str = "latest_checkpoint"):
    """
    Reads `logs/simulation_metrics.txt`, determines the avg_mtt and avg_waiting for `episode`,
    and updates the top-5 lists/saved checkpoints under base_path/TT/1..5 and base_path/WT/1..5.

    The ranking is ascending: lower meanTravelTime is better; lower waitingTime is better.
    We keep lists sorted by ascending value (best first). If a value is None, we ignore updates for that metric.
    """
    metrics = _read_latest_simulation_metrics()
    if episode not in metrics:
        print(f"[update_top5_checkpoints] No metrics found for episode {episode}; skipping top5 update.")
        return

    ep_metrics = metrics[episode]
    _ensure_top_dirs(base_path)
    rankings = _load_rankings(base_path)

    # Helper to update single metric ranking
    def _update_metric(metric_key: str, metric_value: float, folder_name: str):
        if metric_value is None:
            return
        prev_list = list(rankings.get(metric_key, []))
        prev_pos = {e['episode']: idx + 1 for idx, e in enumerate(prev_list)}

        # Remove any existing entry for the same episode (we'll re-insert)
        lst = [e for e in prev_list if e.get('episode') != episode]
        # build entry and append
        entry = {'episode': episode, 'value': metric_value}
        lst.append(entry)
        # sort ascending (lower is better) and keep unique episodes
        lst = sorted(lst, key=lambda x: x['value'])
        # truncate to 5
        lst = lst[:5]
        rankings[metric_key] = lst

        base = Path(base_path)
        metric_parent = base / folder_name
        metric_parent.mkdir(parents=True, exist_ok=True)

        # Move existing slot dirs to a temp area to avoid conflicts during reordering
        tmp_moves = metric_parent / "_tmp_moves"
        if tmp_moves.exists():
            try:
                shutil.rmtree(tmp_moves)
            except Exception:
                pass
        tmp_moves.mkdir(parents=True, exist_ok=True)

        # Move any existing slot directories into tmp_moves with episode-based names
        for i in range(1, 6):
            old_dir = metric_parent / str(i)
            if old_dir.exists():
                # read metadata to know which episode this slot represents (if available)
                meta_ep = None
                meta_file = old_dir / "metadata.json"
                if meta_file.exists():
                    try:
                        meta = json.loads(meta_file.read_text(encoding='utf-8'))
                        meta_ep = meta.get('episode')
                    except Exception:
                        meta_ep = None
                # if we could determine episode, place under tmp_moves/ep_<episode>, else use index
                name = f"ep_{meta_ep}" if meta_ep is not None else f"slot_{i}"
                dest = tmp_moves / name
                try:
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.move(str(old_dir), str(dest))
                except Exception:
                    # best-effort: try to copy then remove
                    try:
                        shutil.copytree(str(old_dir), str(dest))
                        shutil.rmtree(old_dir)
                    except Exception:
                        pass

        # For each target slot, place the appropriate directory
        for slot_idx, slot_entry in enumerate(lst, start=1):
            target_dir = metric_parent / str(slot_idx)
            # If this slot is the current episode, copy from the main latest_checkpoint
            if slot_entry['episode'] == episode:
                # copy all expected files from base_path into target_dir
                if target_dir.exists():
                    try:
                        shutil.rmtree(target_dir)
                    except Exception:
                        pass
                target_dir.mkdir(parents=True, exist_ok=True)
                # copy per-region checkpoint files from main latest_checkpoint
                for region_id in region_controllers.keys():
                    src_ck = base / f"region_{region_id}_checkpoint.pth"
                    src_mem = base / f"region_{region_id}_memory.pkl"
                    if src_ck.exists():
                        try:
                            shutil.copy2(str(src_ck), str(target_dir / src_ck.name))
                        except Exception:
                            pass
                    if src_mem.exists():
                        try:
                            shutil.copy2(str(src_mem), str(target_dir / src_mem.name))
                        except Exception:
                            pass
                # write metadata
                try:
                    (target_dir / "metadata.json").write_text(json.dumps({'episode': slot_entry['episode']}, indent=2), encoding='utf-8')
                except Exception:
                    pass
            else:
                # Try to find a moved directory for this episode in tmp_moves
                moved = tmp_moves / f"ep_{slot_entry['episode']}"
                if moved.exists():
                    # move it into the correct slot position
                    if target_dir.exists():
                        try:
                            shutil.rmtree(target_dir)
                        except Exception:
                            pass
                    try:
                        shutil.move(str(moved), str(target_dir))
                    except Exception:
                        try:
                            shutil.copytree(str(moved), str(target_dir))
                        except Exception:
                            pass
                else:
                    # Fallback: try to copy from main latest_checkpoint if it corresponds to this episode
                    # (only valid if main checkpoint was saved for that episode); otherwise leave empty
                    # We'll attempt to copy if the main metadata.json points to this episode
                    main_meta = base / "metadata.json"
                    copied = False
                    if main_meta.exists():
                        try:
                            mm = json.loads(main_meta.read_text(encoding='utf-8'))
                            if int(mm.get('episode', -1)) == slot_entry['episode']:
                                if target_dir.exists():
                                    try:
                                        shutil.rmtree(target_dir)
                                    except Exception:
                                        pass
                                target_dir.mkdir(parents=True, exist_ok=True)
                                for region_id in region_controllers.keys():
                                    src_ck = base / f"region_{region_id}_checkpoint.pth"
                                    src_mem = base / f"region_{region_id}_memory.pkl"
                                    if src_ck.exists():
                                        try:
                                            shutil.copy2(str(src_ck), str(target_dir / src_ck.name))
                                            copied = True
                                        except Exception:
                                            pass
                                    if src_mem.exists():
                                        try:
                                            shutil.copy2(str(src_mem), str(target_dir / src_mem.name))
                                            copied = True
                                        except Exception:
                                            pass
                                try:
                                    (target_dir / "metadata.json").write_text(json.dumps({'episode': slot_entry['episode']}, indent=2), encoding='utf-8')
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    if not copied:
                        # leave slot empty or write current controllers as last resort
                        if target_dir.exists():
                            try:
                                shutil.rmtree(target_dir)
                            except Exception:
                                pass
                        target_dir.mkdir(parents=True, exist_ok=True)
                        # save current controllers as fallback
                        for region_id, controller in region_controllers.items():
                            checkpoint_filepath = target_dir / f"region_{region_id}_checkpoint.pth"
                            memory_filepath = target_dir / f"region_{region_id}_memory.pkl"
                            torch_state = {
                                'policy_gnn_state_dict': controller.policy_gnn.state_dict(),
                                'target_gnn_state_dict': controller.target_gnn.state_dict(),
                                'policy_net_state_dict': controller.policy_net.state_dict(),
                                'target_net_state_dict': controller.target_net.state_dict(),
                                'optimizer_state_dict': controller.optimizer.state_dict(),
                                'epsilon': controller.epsilon,
                                'update_step': controller.update_step,
                                'gamma': controller.gamma,
                                'epsilon_min': controller.epsilon_min,
                                'epsilon_decay': controller.epsilon_decay,
                                'learning_rate': controller.learning_rate,
                                'target_update': controller.target_update
                            }
                            try:
                                torch.save(torch_state, str(checkpoint_filepath))
                                with open(memory_filepath, 'wb') as f:
                                    pickle.dump(controller.memory, f)
                            except Exception:
                                pass
                        try:
                            (target_dir / "metadata.json").write_text(json.dumps({'episode': slot_entry['episode']}, indent=2), encoding='utf-8')
                        except Exception:
                            pass

        # Cleanup tmp_moves
        try:
            if tmp_moves.exists():
                shutil.rmtree(tmp_moves)
        except Exception:
            pass

    # Update TT (travel time): key "TT" in rankings, metric avg_mtt
    _update_metric('TT', ep_metrics.get('avg_mtt'), 'TT')
    # Update WT (waiting time)
    _update_metric('WT', ep_metrics.get('avg_waiting'), 'WT')

    _save_rankings(rankings, base_path)


def episode_in_top5(episode: int, log_path: str = "logs/simulation_metrics.txt", base_path: str = "latest_checkpoint") -> bool:
    """
    Determine whether the provided episode would enter the TT or WT top-5 lists based on
    the latest metrics file and current rankings. Returns True if the episode should be saved.
    """
    metrics = _read_latest_simulation_metrics(log_path)
    if episode not in metrics:
        return False
    ep_metrics = metrics[episode]
    rankings = _load_rankings(base_path)

    def _qualifies(metric_key: str, val: float) -> bool:
        if val is None:
            return False
        lst = rankings.get(metric_key, [])
        if len(lst) < 5:
            return True
        # worst is the max value in the current top5 (since we store sorted ascending)
        try:
            worst = max(e['value'] for e in lst)
        except Exception:
            return True
        return val < worst

    return _qualifies('TT', ep_metrics.get('avg_mtt')) or _qualifies('WT', ep_metrics.get('avg_waiting'))
