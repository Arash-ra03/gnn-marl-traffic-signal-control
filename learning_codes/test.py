import traci
from Region import RegionController, Actor
from Enviroment import Environment
from typing import List, Dict
from tls_states import GREEN_DURATION, EXTEND_DURATION, YELLOW_DURATION, AVAILABLE_ACTIONS
import random, torch
import numpy as np
import json
from utils.checkpoint_utils import save_checkpoint
from utils.plot_and_summerize_episode import plot_and_summarize_episode
from utils.log_episode_metrics import log_episode_metrics
from utils.actor_logger import log_actor_action
import time


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

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# constants
WINDOW_TIME = 50
NUM_EPISODES = 10
TOTAL_STEPS = 17999
CHECKPOINT_DUMP_FREQ = 10

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

env = Environment(
    sumo_cfg="../5x5.sumocfg",
    regions=regions_dict,
    region_controllers=regions,
)

# Create a DQNAgent for each region
region_controllers = {}
for region_id in env.regions.keys():
    region_controllers[region_id] = regions[region_id]
region_controllers: Dict[int, RegionController]

print("region_controllers dictionary is like this: ")
print(region_controllers)
env.start(region_controllers)
# TRAIN
for ep in range(NUM_EPISODES):
    start_time = time.time()
    # Reset environment
    env.reset(region_controllers)
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
                        reward = env.get_actor_reward(actor.name)
                        log_actor_action(actor.name, "change_logic",
                                         _create_actor_log_details(actor=actor, reward=reward, ep=ep))


                    else:
                        state = actor.state
                        action = actor.current_action
                        reward = env.get_actor_reward(actor.name)
                        episode_reward += reward  # accumaltive reward
                        next_state = env.get_agent_state(actor.name, actor.current_action)
                        region_controller.store_transition(state=state, action=action, reward=reward,
                                                           next_state=next_state, done=False, actor_type=actor.type)

                        state = next_state
                        actor.state = state
                        action = region_controller.choose_action_for_junction(actor.state, actor.type)

                        if action in AVAILABLE_ACTIONS[actor.current_action]:
                            actor.is_yellow = False
                            actor.current_action = action
                            actor.tls_state = traci.trafficlight.getProgram(actor.name)
                            reward = env.get_actor_reward(actor.name)
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
                            reward = env.get_actor_reward(actor.name)
                            log_actor_action(actor.name, "set_yellow_logic",
                                             _create_actor_log_details(actor=actor, reward=reward, ep=ep))





        traci.simulationStep()
        current_step += 1
        current_time = traci.simulation.getTime()

        if current_step % 20 == 0:
            for region_id, region_controller in region_controllers.items():
                loss_value = region_controller.train_step()
                region_reward = env.get_region_reward(region_id)
                log_entry = {}  # training info log dict
                if loss_value is not None:
                    log_entry = _create_training_info_log_entry(region_controller, region_reward, loss_value, ep)
                else:
                    # Not enough memory for a batch yet; still log something
                    log_entry = _create_training_info_log_entry(region_controller, region_reward, "NotEnoughMemory", ep)

                region_controller.region_logger.info(json.dumps(log_entry))
                region_controller.episode_data.append(log_entry)
        if traci.simulation.getTime() >= TOTAL_STEPS:
            done = True
    plot_and_summarize_episode(ep + 1, region_controllers, NUM_EPISODES)
    # Decay epsilon for each agent
    for region_id, agent in region_controllers.items():
        agent.update_epsilon()

    print(f"Episode {ep + 1}/{NUM_EPISODES}, Reward: {episode_reward}")

    end_time = time.time()
    runtime_log_msg = _calc_episode_runtime(end_time, start_time)
    log_episode_metrics(ep + 1, runtime_log_msg)  # <-- call here

    # Save and overwrite a checkpoint every 10 episodes
    #
    if (ep + 1) % CHECKPOINT_DUMP_FREQ == 0:
        save_checkpoint(region_controllers, ep + 1)

print("Training finished.")
env.close()