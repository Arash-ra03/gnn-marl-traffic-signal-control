import os
import random
import numpy as np
from collections import deque

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# SUMO/TraCI imports
import traci

# Gym-like environment base
import gym
from gym import spaces


###############################################################################
# 1. ENVIRONMENT DEFINITION
###############################################################################
class TrafficJunctionEnv(gym.Env):
    """
    A Gym environment for controlling traffic lights in multiple regions.

    Each region is a set of junctions (traffic lights).
    The environment expects a dict of actions {region_id: [actions_for_junctions]}.

    The state is similarly a dict: {region_id: [states_of_junctions]}.
    Each junction's state is [vehicles_vertical, vehicles_horizontal, junction_type].
    """

    def __init__(self, sumo_cfg, regions, region_type='4-way', max_time=60):
        """
        :param sumo_cfg: Path to the SUMO configuration file (.sumocfg).
        :param regions: A dictionary mapping region_id -> list of junction IDs.
                        Example: {0: ['A0','B0','C0'], 1: ['A1','B1','C1']}
        :param region_type: '3-way' or '4-way', just for storing in the state array.
        :param max_time: Maximum green time increment/decrement.
        """
        super(TrafficJunctionEnv, self).__init__()

        # Environment config
        self.sumo_cfg = sumo_cfg
        self.regions = regions
        self.region_type = region_type
        self.max_time = max_time
        self.time_step = 1  # time step in seconds per simulation step
        self.current_step = 0
        self.max_steps = 100  # you can adjust this

        # Start SUMO (will be restarted in reset())
        # You can pass additional arguments to `traci.start()` if needed,
        # e.g., for a GUI version: ["sumo-gui", "-c", sumo_cfg]
        traci.start(["sumo", "-c", sumo_cfg])

        # Action space: 3 discrete actions per junction (decrease/hold/increase)
        # The environment is multi-agent, but we define a single action space for each region's agent:
        #   0 -> decrease green time
        #   1 -> hold
        #   2 -> increase green time
        self.action_space = spaces.Discrete(3)

        # Observation space: per junction [vehicles_vertical, vehicles_horizontal, junction_type]
        # We'll treat each junction as a (3,) Box. Because we store states in a dict {region_id -> [list of states]},
        # we won't define a single Box shape for the entire environment, but for *one* junction:
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)

        # Dictionary to hold states for each region
        self.region_states = {}

    def reset(self):
        """
        Resets the SUMO simulation and re-initializes the state for each junction.
        Returns a dict of states: {region_id: [ [v_vert, v_horz, type], ..., ] }
        """
        # Reload the simulation from the beginning
        traci.load(["-c", self.sumo_cfg])
        self.current_step = 0
        self.region_states = {}

        # Gather initial states
        for region_id, junctions in self.regions.items():
            region_state = []
            for junction_id in junctions:
                vehicles_vertical = self._get_vehicles_count(junction_id, "vertical")
                vehicles_horizontal = self._get_vehicles_count(junction_id, "horizontal")
                junction_type = 4 if self.region_type == '4-way' else 3
                region_state.append([vehicles_vertical, vehicles_horizontal, junction_type])
            self.region_states[region_id] = region_state

        return self.region_states

    def step(self, region_actions):
        """
        region_actions: dict {region_id: [a_0, a_1, ..., a_n]}, where each a_i in {0,1,2}
        (decrease, hold, increase) for that region's junctions.
        """
        # Apply actions and compute rewards for each region
        next_state = {}
        rewards = {}

        for region_id, actions in region_actions.items():
            region_reward = 0
            next_region_state = []

            junctions = self.regions[region_id]
            for i, action in enumerate(actions):
                junction_id = junctions[i]
                # Current state of that junction
                v_vert, v_horz, j_type = self.region_states[region_id][i]

                # Decide how much to change the traffic light time
                if action == 0:  # decrease
                    light_time_change = -5
                elif action == 1:  # hold
                    light_time_change = 0
                elif action == 2:  # increase
                    light_time_change = 5

                # Apply the action to the traffic light. Here we do something simplistic:
                # We just add/sub the change to the current phase index (phase cycles).
                current_phase = traci.trafficlight.getPhase(junction_id)
                new_phase = (current_phase + light_time_change) % traci.trafficlight.getPhaseNumber(junction_id)
                traci.trafficlight.setPhase(junction_id, new_phase)

                # Get the new vehicle counts
                updated_v_vert = self._get_vehicles_count(junction_id, "vertical")
                updated_v_horz = self._get_vehicles_count(junction_id, "horizontal")

                # The reward is negative of the halted vehicles
                halted_vehicles = updated_v_vert + updated_v_horz
                reward = -halted_vehicles
                region_reward += reward

                # Construct next state for this junction
                next_region_state.append([updated_v_vert, updated_v_horz, j_type])

            next_state[region_id] = next_region_state
            rewards[region_id] = region_reward

        # Advance the SUMO simulation by one step
        traci.simulationStep()
        self.current_step += 1
        self.region_states = next_state

        # We consider an episode done after self.max_steps timesteps
        done = (self.current_step >= self.max_steps)

        return next_state, rewards, done, {}

    def render(self, mode='human'):
        """
        Basic rendering: print out the region states.
        """
        print("Current states:", self.region_states)

    def close(self):
        """
        Close the SUMO simulation.
        """
        traci.close()

    def _get_vehicles_count(self, junction_id, lane_name_suffix):
        """
        Helper to fetch the number of vehicles from an induction loop or lane near a junction.
        This is just an example. You need to adapt to how your SUMO network is set up (e.g., loop IDs).

        If you have loops named like 'A0_vertical', 'A0_horizontal', etc., you can adapt to:
            loop_id = f"{junction_id}_{lane_name_suffix}"
            return traci.inductionloop.getLastStepVehicleNumber(loop_id)

        Alternatively, if you track them via lane-based queries:
            lanes = traci.trafficlight.getControlledLanes(junction_id)
            # filter vertical/horizontal lanes, etc.

        For demonstration, we return a random number here (if you don't have real loops).
        """
        # Example using induction loops named <junctionID>_<lane>:
        loop_id = f"{junction_id}_{lane_name_suffix}"
        try:
            count = traci.inductionloop.getLastStepVehicleNumber(loop_id)
        except traci.exceptions.TraCIException:
            # If loop doesn't exist, fallback to random or zero for demonstration
            count = random.randint(0, 20)
        return count


###############################################################################
# 2. NEURAL NETWORK / DQN DEFINITION
###############################################################################
class DQN(nn.Module):
    """
    Simple feed-forward network for Q-value approximation.
    """

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    """
    A DQN agent controlling one region. It takes the region's state (list of junction states),
    flattens them, and outputs action-values for each junction in that region.
    """

    def __init__(self, env, region_id, lr=1e-3, gamma=0.99, batch_size=32):
        self.env = env
        self.region_id = region_id

        # Number of junctions in this region
        self.num_junctions = len(env.regions[region_id])
        # Each junction state is of dimension 3: [vehicles_vert, vehicles_horz, j_type]
        self.state_dim = 3 * self.num_junctions  # flatten them
        # Each junction has 3 possible actions, so total = 3 * num_junctions
        self.action_dim = 3 * self.num_junctions

        # Q networks
        self.model = DQN(self.state_dim, self.action_dim)
        self.target_model = DQN(self.state_dim, self.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=10000)

        # Epsilon-greedy
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        # Update frequency for target network
        self.update_target_every = 10
        self.update_step = 0

    def get_action(self, region_state):
        """
        region_state: list of length `num_junctions`, each [v_vert, v_horz, j_type].
        We flatten it to shape (state_dim,) then feed to the DQN.

        Output: a list of actions (one for each junction).
        """
        # Epsilon-greedy
        if random.random() < self.epsilon:
            # Random actions for each junction
            return [random.randint(0, 2) for _ in range(self.num_junctions)]
        else:
            # Exploit
            state_tensor = self._to_tensor(region_state).unsqueeze(0)  # shape (1, state_dim)
            with torch.no_grad():
                q_values = self.model(state_tensor)  # shape (1, action_dim)
            q_values = q_values[0].view(self.num_junctions, 3)  # reshape to (num_junctions, 3)
            actions = torch.argmax(q_values, dim=1).tolist()
            return actions

    def store_transition(self, state, action_list, reward, next_state, done):
        """
        Store experience in replay buffer. We'll convert (state, action_list) -> flatten.
        """
        self.memory.append((state, action_list, reward, next_state, done))

    def train_step(self):
        """
        Sample from memory, update the model (one step of gradient descent).
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        # Prepare arrays/tensors
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for s, a, r, ns, d in batch:
            states.append(self._to_tensor(s))
            actions.append(a)
            rewards.append(r)
            next_states.append(self._to_tensor(ns))
            dones.append(float(d))

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # actions is a list of lists. Flatten or transform to indices
        # For each junction, we have an action in {0,1,2}, total is 3 * num_junctions
        # But we selected exactly one action per junction -> we must compute the Q-value index.
        # Example: if action_list = [2,0] for 2 junctions, that's action index 2 for the first junction
        # and 0 for the second, so overall indexes: [2, 3*(2)+0] ? Actually we do them in separate Q-values.
        #
        # We'll handle this by:
        #   - Reshape Q-values to (batch_size, num_junctions, 3).
        #   - Then gather for each junction the action's Q-value, sum or average them.
        #
        # Alternatively, we can store them as a single combined index, but that's complicated.
        # We'll do the sum of Q-values for the chosen actions across all junctions.

        # Evaluate current Q(s) from online model
        q_values = self.model(states)  # (batch_size, action_dim)
        # Reshape to (batch_size, num_junctions, 3)
        q_values = q_values.view(-1, self.num_junctions, 3)

        # Evaluate next Q(s') from target model
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            next_q_values = next_q_values.view(-1, self.num_junctions, 3)

        # Gather chosen Q-values and next Q-values
        chosen_q_list = []
        max_next_q_list = []

        for i in range(self.batch_size):
            chosen_q_sum = 0.0
            max_q_sum = 0.0
            for j in range(self.num_junctions):
                action_j = actions[i][j]  # each junction's action
                chosen_q_sum += q_values[i, j, action_j].item()
                max_q_sum += torch.max(next_q_values[i, j]).item()
            chosen_q_list.append(chosen_q_sum)
            max_next_q_list.append(max_q_sum)

        chosen_q_tensor = torch.tensor(chosen_q_list, dtype=torch.float32)
        max_next_q_tensor = torch.tensor(max_next_q_list, dtype=torch.float32)

        # Bellman update
        targets = rewards + self.gamma * max_next_q_tensor * (1 - dones)

        loss = nn.MSELoss()(chosen_q_tensor, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Increment step counter for target network updates
        self.update_step += 1
        if self.update_step % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        """
        Decays epsilon after each episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def _to_tensor(self, region_state):
        """
        region_state is a list of lists: [[v_vert, v_horz, j_type], ...]
        Flatten into a 1D array.
        """
        flat_list = []
        for s in region_state:
            flat_list.extend(s)  # s is [v_vert, v_horz, j_type]
        return torch.tensor(flat_list, dtype=torch.float32)


###############################################################################
# 3. MULTI-REGION MANAGER
###############################################################################
class MultiRegionDQNTrainer:
    """
    Manages multiple DQN agents (one for each region), training them together
    in a single environment step.
    """

    def __init__(self, env):
        self.env = env
        # Create a DQNAgent for each region
        self.agents = {}
        for region_id in env.regions.keys():
            self.agents[region_id] = DQNAgent(env, region_id)

    def train(self, episodes=1000):
        """
        Train all region agents together for a specified number of episodes.
        """
        for ep in range(episodes):
            # Reset environment
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Gather actions from each agent
                actions_dict = {}
                for region_id, agent in self.agents.items():
                    region_state = state[region_id]
                    region_action_list = agent.get_action(region_state)
                    actions_dict[region_id] = region_action_list

                # Step the environment with all region actions
                next_state, rewards_dict, done, _ = self.env.step(actions_dict)

                # Store transitions in each agent's memory
                for region_id, agent in self.agents.items():
                    agent.store_transition(
                        state[region_id],
                        actions_dict[region_id],
                        rewards_dict[region_id],
                        next_state[region_id],
                        done
                    )

                # Train each agent on sampled experiences
                for region_id, agent in self.agents.items():
                    agent.train_step()

                state = next_state
                # Sum the total reward from all regions (optional, for logging)
                episode_reward += sum(rewards_dict.values())

            # Decay epsilon for each agent
            for region_id, agent in self.agents.items():
                agent.update_epsilon()

            print(f"Episode {ep + 1}/{episodes}, Reward: {episode_reward}")

        print("Training finished.")
        self.env.close()


###############################################################################
# 4. EXAMPLE MAIN
###############################################################################
if __name__ == "__main__":
    # Path to your SUMO configuration file
    sumo_cfg_path = "path/to/your/sumo_config.sumocfg"

    # Define regions and their junctions
    # Example: region 0 -> [A0, B0, C0], region 1 -> [A1, B1, C1]
    regions_dict = {
        0: ['A0', 'B0', 'C0'],
        1: ['A1', 'B1', 'C1']
    }

    # Create environment
    env = TrafficJunctionEnv(
        sumo_cfg=sumo_cfg_path,
        regions=regions_dict,
        region_type='4-way',
        max_time=60
    )

    # Create multi-region trainer (one agent per region)
    multi_region_trainer = MultiRegionDQNTrainer(env)

    # Train
    multi_region_trainer.train(episodes=50)  # Example: 50 episodes
