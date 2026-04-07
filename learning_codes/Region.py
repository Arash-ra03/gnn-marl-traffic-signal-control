import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from collections import namedtuple
import random

from tls_states import GREEN_DURATION
from tls_states import VALID_ACTIONS
from DQN import DQN
from typing import NamedTuple, List
import logging, os, json

DIM_STATE = 9 + 8 + 2 + (4 * 13)
N_ACTION = 4*2
INIT_TLS_STATE = "NS_S"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MEMORY_REP_SIZE = 50000
BATCH_SIZE = 64


class SingleTransition(NamedTuple):
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool
    actor_type: str


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(SingleTransition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# contains metadata of each single actor
class Actor:
    def __init__(self, name, type_, tls_state, next_time, current_action, state, is_yellow):
        self.name = name
        self.type = type_
        self.tls_state = tls_state
        self.next_time = next_time
        self.current_action = current_action
        self.state = state
        self.is_yellow = is_yellow

# next_time stores next time that actor should choose its next action
# is_yellow shows actor is in yellow phase or not


class RegionController:
    def __init__(self, region_id, agent_ids: List[str]):
        # ADD the following lines to create a logger for this region
        log_dir = "logs/training_logs"
        os.makedirs(log_dir, exist_ok=True)
        logger_name = f"region_{region_id}"
        self.region_logger = logging.getLogger(logger_name)
        self.region_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{logger_name}.log"), mode="w")
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        self.region_logger.addHandler(file_handler)
        # Prevent duplication if root logger is also configured
        self.region_logger.propagate = False
        self.episode_data = []

        # TODO Revise setting inital tls_state
        # self.agents = agents
        self.actors = []
        self.region_id = region_id
        self.num_junctions = len(agent_ids)


        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.990  # Decay factor per episode
        self.learning_rate = 0.00025
        self.target_update = 1000  # Episodes between target network updates

        self.policy_net = DQN(DIM_STATE, N_ACTION).to(device)
        self.target_net = DQN(DIM_STATE, N_ACTION).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in evaluation mode

        self.update_step = 0 # Tracks the number of training steps performed

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(MEMORY_REP_SIZE)

        #TODO initialize next_time and chosen_action
        for i in range(len(agent_ids)):
            if agent_ids[i][0] == "A":
                self.actors.append(
                    Actor(name=agent_ids[i], type_="3_A", tls_state="NS_S", next_time= GREEN_DURATION, current_action= None, state = None, is_yellow = False)
                )
            elif agent_ids[i][0] == "E":
                self.actors.append(
                    Actor(name=agent_ids[i], type_="3_E", tls_state="NS_S", next_time= GREEN_DURATION, current_action = None, state = None, is_yellow = False)
                )
            elif agent_ids[i][1] == "4":
                self.actors.append(
                    Actor(name=agent_ids[i], type_="3_4", tls_state="NS_S", next_time= GREEN_DURATION, current_action = None, state = None, is_yellow = False)
                )
            elif agent_ids[i][1] == "0":
                self.actors.append(
                    Actor(name=agent_ids[i], type_="3_0", tls_state="NS_S", next_time= GREEN_DURATION, current_action = None, state = None, is_yellow = False)
                )
            else:
                self.actors.append(
                    Actor(name=agent_ids[i], type_="4", tls_state="NS_S", next_time= GREEN_DURATION, current_action = None, state = None, is_yellow = False)
                )



    def choose_action(self, region_state: List[List]) -> List:
        if random.random() < self.epsilon:
            # Random actions for each junction
            actions = []
            for actor in self.actors:
                actor:Actor
                # Random actions for each junction
                valid_actions = VALID_ACTIONS[actor.type]
                action = random.choice(valid_actions)
                actions.append(action)
            return actions
        else:
            # Convert the 2D list of actor states to a tensor of shape (num_actors, state_dim)
            state_tensor = self._to_tensor(region_state)
            with torch.no_grad():
                # The policy network returns a tensor of shape (num_actors, num_actions)
                q_values = self.policy_net(state_tensor)

            actions = []
            for i, actor in enumerate(self.actors):
                actor: Actor
                valid_actions = VALID_ACTIONS[actor.type]

                # Get Q-values for this actor shape((num_actions, ))
                q_values_for_actor = q_values[i] 

                # Create a mask for invalid actions
                mask = torch.full(q_values_for_actor.shape, float('-inf'))
                mask[valid_actions] = 0  # Set valid actions to 0, keeping their values

                # Apply the mask
                masked_q_values = q_values_for_actor + mask

                # Choose action with the highest Q-value
                action = torch.argmax(masked_q_values).item()
                actions.append(action)

            return actions

    def choose_action_for_junction(self, single_state: List[float], actor_type: str) -> int:
        """
        Decide an action for one junction, given its single-state vector and its type.
        Returns the chosen action (integer).
        """
        # With probability epsilon, choose a random valid action
        if random.random() < self.epsilon:
            valid_actions = VALID_ACTIONS[actor_type]
            return random.choice(valid_actions)
        else:
            # Convert the single-state to a tensor of shape (1, state_dim)
            state_tensor = torch.tensor([single_state], dtype=torch.float32, device=device)

            # Forward pass through the policy network
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                # q_values will have shape (1, num_actions)

            q_values_for_junction = q_values[0]  # shape: (num_actions, )

            # Mask out invalid actions
            mask = torch.full(q_values_for_junction.shape, float('-inf')).to(device)
            valid_actions = VALID_ACTIONS[actor_type]
            mask[valid_actions] = 0.0

            # Apply the mask
            masked_q_values = q_values_for_junction + mask

            # Choose the argmax
            action = torch.argmax(masked_q_values).item()
            return action

    def _to_tensor(self, region_state: List[List]) -> torch.Tensor:
        """Helper function to convert region state list to tensor for the model."""
        return torch.tensor(region_state, dtype=torch.float32)

    def store_transition(self,
                         state: List[float],
                         action: int,
                         reward: float,
                         next_state: List[float],
                         done: bool,
                         actor_type: str):
        """Store ONE junction's transition."""
        self.memory.push(state, action, reward, next_state, done,actor_type)

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)

        # Now each item in 'batch' is a SingleTransition
        # We'll gather them into arrays
        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []
        dones_list = []
        actor_types_list = []

        for transition in batch:
            states_list.append(torch.tensor(transition.state, dtype=torch.float32))
            actions_list.append(transition.action)
            rewards_list.append(transition.reward)
            next_states_list.append(torch.tensor(transition.next_state, dtype=torch.float32))
            dones_list.append(1.0 if transition.done else 0.0)
            actor_types_list.append(transition.actor_type)

        # Stack into tensors
        states_tensor = torch.stack(states_list).to(device)  # shape (B, state_dim)
        next_states_tensor = torch.stack(next_states_list).to(device)  # shape (B, state_dim)

        actions_tensor = torch.tensor(actions_list, dtype=torch.long).to(device)
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32).to(device)
        dones_tensor = torch.tensor(dones_list, dtype=torch.float32).to(device)

        # Forward pass for current Q
        current_q_values = self.policy_net(states_tensor)  # shape (B, num_actions)
        # Gather the Q-value for each chosen action - already masked so its valid
        chosen_q = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Next Q (with Action Masking for Double DQN)
        with torch.no_grad():
            # 1. Get the Q-values for the next states from the policy network
            next_q_values_policy = self.policy_net(next_states_tensor) # Shape: (B, N_ACTION)

            # 2. Create a mask for the batch based on each transition's actor_type
            batch_mask = torch.full_like(next_q_values_policy, float('-inf'))
            for i, actor_type in enumerate(actor_types_list):
                valid_actions = VALID_ACTIONS[actor_type]
                batch_mask[i, valid_actions] = 0.0

            # 3. Apply the mask to find the best *valid* actions
            masked_next_q_values = next_q_values_policy + batch_mask
            best_actions = masked_next_q_values.argmax(dim=1)

            # 4. Use the target network to get the Q-value of these best actions
            next_q_values_target = self.target_net(next_states_tensor)
            max_next_q = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            # next_q_values = self.target_net(next_states_tensor)
            # max_next_q, _ = next_q_values.max(dim=1)

        target_q = rewards_tensor + self.gamma * max_next_q * (1 - dones_tensor)

        # Compute loss
        loss = nn.MSELoss()(chosen_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_step += 1
        if self.update_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def update_epsilon(self):
        """
        Decays epsilon after each episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
