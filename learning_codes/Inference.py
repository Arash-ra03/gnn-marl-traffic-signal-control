import traci
from Region import RegionController, Actor
from Enviroment import Environment
from typing import List, Dict
from tls_states import GREEN_DURATION, EXTEND_DURATION, YELLOW_DURATION, SOFT_TRANSITION
import random, torch
import numpy as np
import json
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.plot_and_summerize_episode import plot_and_summarize_episode
from utils.log_episode_metrics import log_episode_metrics
from utils.actor_logger import log_actor_action
import time
import os
import shutil


def _create_actor_log_details(actor: Actor, reward, ep) -> Dict:
    """
        Used to check healthy execution of each actor.
    """
    return {
        "episode": ep + 1,
        "simulation_time": traci.simulation.getTime(),
        "tls_state": actor.tls_state,
        "next_time": actor.next_time,
        "current_action": actor.current_action,
        "state": actor.state,
        "is_yellow": actor.is_yellow,
        "reward": reward
    }


def _create_training_info_log_entry(region_controller: RegionController, region_reward, loss_value, ep) -> Dict:
    """
        Training Info for each RegionController per episode.
    """
    return {
        "episode": ep + 1,
        "simulation_time": traci.simulation.getTime(),
        "train_step": region_controller.update_step,
        "region_reward": region_reward,
        "loss": loss_value
    }


def _calc_episode_runtime(end_time, start_time):
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    return f"{minutes} minute(s) and {seconds:.2f} second(s)"


def _is_extend(action):
    if action >= 4:
        return True
    else:
        return False


def save_episode_outputs(ep_num: int):
    """
    Copy SUMO output files from the repository `outputs` folder into
    `outputs/episode_<ep_num>/` to avoid overwrites between episodes.

    Files copied (if present): tripinfo.xml, summary.xml, statistics.xml,
    queue.xml, queue_length.csv
    """
    # outputs folder is at repository root relative to this file
    repo_outputs = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    episode_dir = os.path.join(repo_outputs, f"episode_{ep_num}")
    try:
        os.makedirs(episode_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create episode directory '{episode_dir}': {e}")
        return

    filenames = ["tripinfo.xml", "summary.xml", "statistics.xml", "queue.xml", "queue_length.csv"]
    for fname in filenames:
        src = os.path.join(repo_outputs, fname)
        if os.path.exists(src):
            dst = os.path.join(episode_dir, fname)
            # avoid accidental overwrite if same episode dir already has file
            if os.path.exists(dst):
                base, ext = os.path.splitext(fname)
                dst = os.path.join(episode_dir, f"{base}_{int(time.time())}{ext}")
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Failed to copy {src} -> {dst}: {e}")


# constants
WINDOW_TIME = 50
NUM_EPISODES = 410
TOTAL_STEPS = 17999
CHECKPOINT_DUMP_FREQ = 10
SEED = 42

regions_dict = {
    0: ['A1', 'A2', 'A3'],
    1: ['B0', 'B1', 'C0', 'C1', 'D0', 'D1'],
    2: ['B3', 'B4', 'C3', 'C4', 'D3', 'D4'],
    3: ['E1', 'E2', 'E3'],
    4: ['B2', 'C2', 'D2'],
}

regions = [RegionController(reg, regions_dict[reg]) for reg in regions_dict]
regions: List[RegionController]
regions_dict = {
    0: regions[0].actors,
    1: regions[1].actors,
    2: regions[2].actors,
    3: regions[3].actors,
    4: regions[4].actors,
}

same_region_onehop_neighbors = {
    'A1': ["A2", None, None, None], 'A2': ["A3", None, "A1", None], 'A3': [None, None, "A2", None],
    'B0': ["B1", "C0", None, None], 'B1': [None, "C1", "B0", None], 'C0': ["C1", "D0", None, "B0"],
    'C1': [None, "D1", "C0", "B1"], 'D0': ["D1", None, None, "C0"], 'D1': [None, None, "D0", "C1"],
    'B3': ["B4", "C3", None, None], 'B4': [None, "C4", "B3", None], 'C3': ["C4", "D3", None, "B3"],
    'C4': [None, "D4", "C3", "B4"], 'D3': ["D4", None, None, "C3"], 'D4': [None, None, "D3", "C4"],
    'E1': ["E2", None, None, None], 'E2': ["E3", None, "E1", None], 'E3': [None, None, "E2", None],
    'B2': [None, "C2", None, None], 'C2': [None, "D2", None, "B2"], 'D2': [None, None, None, "C2"]
}
for region_controller in regions:
    region_controller.set_graph(same_region_onehop_neighbors)

env = Environment(
    sumo_cfg="../5x5.sumocfg",
    regions=regions_dict,
    region_controllers=regions,
    _same_region_onehop_neighbors=same_region_onehop_neighbors
)

# Create a DQNAgent for each region
region_controllers = {}
for region_id in env.regions.keys():
    region_controllers[region_id] = regions[region_id]
region_controllers: Dict[int, RegionController]

print("region_controllers dictionary is like this: ")
print(region_controllers)
resume_info = load_checkpoint(region_controllers, path="latest_checkpoint/WT/1")


if resume_info:
    START_EPISODE = resume_info["next_episode"]  # 1-based
else:
    START_EPISODE = 1

print(f"Starting training from episode {START_EPISODE}")
env.start(region_controllers)
# TRAIN
for ep in range(START_EPISODE - 1, NUM_EPISODES):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    SEED += 1
    start_time = time.time()
    # Reset environment
    env.reset(region_controllers)
    for region_controller in region_controllers.values():
        region_local_state = env.get_region_local_states(region_controller)
        for actor in region_controller.actors:
            # keep actor.state as the handcrafted full state
            actor.state = env.get_agent_state(actor.name, actor.current_action)
            actor.region_local_state = [row[:] for row in region_local_state]

    done = False  # episode done
    episode_reward = 0
    current_step = 0

    end_time = traci.simulation.getEndTime()
    print("End time (seconds):", end_time)
    while not done:  # single episode simulation

        for num, region_controller in region_controllers.items():
            for actor in region_controller.actors:
                if current_step >= actor.next_time:

                    if actor.is_yellow:
                        if _is_extend(actor.current_action):
                            env.extend_logic(actor.name, actor.type, actor.current_action)
                            actor.next_time = current_step + EXTEND_DURATION
                        else:
                            env.change_logic(actor.name, actor.type, actor.current_action)
                            actor.next_time = current_step + GREEN_DURATION

                        actor.is_yellow = False
                        actor.tls_state = traci.trafficlight.getProgram(actor.name)
                        # LOG
                        reward = env.get_actor_reward_onehop_centralized(actor.name)
                        log_actor_action(actor.name, "change_logic",
                                         _create_actor_log_details(actor=actor, reward=reward, ep=ep))


                    else:
                        state = actor.state
                        action = actor.current_action
                        reward = env.get_actor_reward_onehop_centralized(actor.name)
                        episode_reward += reward

                        next_state = env.get_agent_state(actor.name, actor.current_action)
                        next_region_local_state = env.get_region_local_states(region_controller)
                        actor_index = region_controller.agent_id_to_idx[actor.name]

                        actor.state = next_state
                        actor.region_local_state = [row[:] for row in next_region_local_state]

                        action = region_controller.choose_action_for_junction(
                            actor.state,
                            actor.type,
                            actor.region_local_state,
                            actor_index
                        )

                        if action in SOFT_TRANSITION[actor.current_action]:
                            actor.is_yellow = False
                            actor.current_action = action
                            actor.tls_state = traci.trafficlight.getProgram(actor.name)
                            reward = env.get_actor_reward_onehop_centralized(actor.name)
                            if _is_extend(action):
                                env.extend_logic(actor.name, actor.type, action)
                                actor.next_time = current_step + EXTEND_DURATION
                                log_actor_action(actor.name, "Extend",
                                                 _create_actor_log_details(actor=actor, reward=reward, ep=ep))
                            else:
                                env.change_logic(actor.name, actor.type, action)
                                actor.next_time = current_step + GREEN_DURATION
                                log_actor_action(actor.name, "Full",
                                                 _create_actor_log_details(actor=actor, reward=reward, ep=ep))



                        else:
                            env.set_yellow_logic(actor.name, actor.type, actor.current_action)
                            actor.is_yellow = True
                            actor.next_time = current_step + YELLOW_DURATION
                            actor.current_action = action
                            actor.tls_state = traci.trafficlight.getProgram(actor.name)
                            # LOG
                            reward = env.get_actor_reward_onehop_centralized(actor.name)
                            log_actor_action(actor.name, "set_yellow_logic",
                                             _create_actor_log_details(actor=actor, reward=reward, ep=ep))

        traci.simulationStep()
        current_step += 1
        current_time = traci.simulation.getTime()

        if traci.simulation.getTime() >= TOTAL_STEPS:
            done = True

    # Save SUMO outputs for this episode so they don't get overwritten
    try:
        save_episode_outputs(ep + 1)
    except Exception as e:
        print(f"Error while saving episode outputs for episode {ep + 1}: {e}")

    print(f"Episode {ep + 1}/{NUM_EPISODES}, Reward: {episode_reward}")

    end_time = time.time()
    runtime_log_msg = _calc_episode_runtime(end_time, start_time)
    log_episode_metrics(ep + 1, runtime_log_msg)  # <-- call here

print("Training finished.")
env.close()