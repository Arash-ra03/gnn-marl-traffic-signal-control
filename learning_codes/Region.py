import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from collections import namedtuple
import random

from tls_states import GREEN_DURATION
from tls_states import VALID_ACTIONS
from DQN import DQN
from GNNEncoder import GNNEncoder
from typing import NamedTuple, List
import logging, os, json

OLD_STATE_DIM = 9 + 8 + 2 + (4 * 13)   # existing handcrafted full state
LOCAL_STATE_DIM = 4 + 5 + 8            # local-only state for GNN nodes
GNN_HIDDEN_DIM = 16
DIM_STATE = OLD_STATE_DIM + GNN_HIDDEN_DIM
N_ACTION = 4 * 2
INIT_TLS_STATE = "NS_S"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MEMORY_REP_SIZE = 50000
BATCH_SIZE = 64


class SingleTransition(NamedTuple):
    state: List[float]                    # old handcrafted full state
    action: int
    reward: float
    next_state: List[float]               # old handcrafted next full state
    done: bool
    actor_type: str
    region_local_state: List[List[float]]
    next_region_local_state: List[List[float]]
    actor_index: int


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
    def __init__(self, name, type_, tls_state, next_time, current_action, state, is_yellow, region_local_state=None):
        self.name = name
        self.type = type_
        self.tls_state = tls_state
        self.next_time = next_time
        self.current_action = current_action
        self.state = state  # old handcrafted full state
        self.is_yellow = is_yellow
        self.region_local_state = region_local_state

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

        self.policy_gnn = GNNEncoder(LOCAL_STATE_DIM, GNN_HIDDEN_DIM).to(device)
        self.target_gnn = GNNEncoder(LOCAL_STATE_DIM, GNN_HIDDEN_DIM).to(device)

        self.policy_net = DQN(DIM_STATE, N_ACTION).to(device)
        self.target_net = DQN(DIM_STATE, N_ACTION).to(device)

        self.target_gnn.load_state_dict(self.policy_gnn.state_dict())
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_gnn.eval()
        self.target_net.eval()

        self.update_step = 0

        self.optimizer = optim.Adam(
            list(self.policy_gnn.parameters()) + list(self.policy_net.parameters()),
            lr=self.learning_rate
        )
        self.memory = ReplayMemory(MEMORY_REP_SIZE)

        self.agent_ids = agent_ids
        self.agent_id_to_idx = {aid: idx for idx, aid in enumerate(agent_ids)}
        self.edge_index = None

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

    def set_graph(self, neighbors_map):
        self.edge_index = self._build_edge_index(neighbors_map).to(device)

    def _build_edge_index(self, neighbors_map):
        edges = []

        for src in self.agent_ids:
            src_idx = self.agent_id_to_idx[src]
            for dst in neighbors_map[src]:
                if dst is None:
                    continue
                if dst not in self.agent_id_to_idx:
                    continue
                dst_idx = self.agent_id_to_idx[dst]
                edges.append([src_idx, dst_idx])

        for i in range(len(self.agent_ids)):
            edges.append([i, i])  # self-loops

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _build_augmented_state(self, single_state, region_local_state, actor_index, use_target=False):
        if self.edge_index is None:
            raise ValueError("edge_index is not set. Call set_graph(...) first.")

        region_local_tensor = torch.tensor(region_local_state, dtype=torch.float32, device=device)

        if use_target:
            gnn_out = self.target_gnn(region_local_tensor, self.edge_index)
        else:
            gnn_out = self.policy_gnn(region_local_tensor, self.edge_index)

        gnn_embed = gnn_out[actor_index]  # [GNN_HIDDEN_DIM]

        single_state_tensor = torch.tensor(single_state, dtype=torch.float32, device=device)
        augmented_state = torch.cat([single_state_tensor, gnn_embed], dim=0)  # [OLD_STATE_DIM + GNN_HIDDEN_DIM]

        return augmented_state.unsqueeze(0)  # [1, DIM_STATE]

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
            state_tensor = self._to_tensor(region_state).to(device)
            with torch.no_grad():
                q_values = self._forward_q(state_tensor, use_target=False)

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

    def choose_action_for_junction(self, single_state: List[float], actor_type: str, region_local_state,
                                   actor_index: int) -> int:
        if random.random() < self.epsilon:
            valid_actions = VALID_ACTIONS[actor_type]
            return random.choice(valid_actions)

        with torch.no_grad():
            state_tensor = self._build_augmented_state(
                single_state=single_state,
                region_local_state=region_local_state,
                actor_index=actor_index,
                use_target=False
            )
            q_values = self.policy_net(state_tensor)

        q_values_for_junction = q_values[0]

        mask = torch.full(q_values_for_junction.shape, float('-inf'), device=device)
        valid_actions = VALID_ACTIONS[actor_type]
        mask[valid_actions] = 0.0

        masked_q_values = q_values_for_junction + mask
        return torch.argmax(masked_q_values).item()

    def _to_tensor(self, region_state: List[List]) -> torch.Tensor:
        """Helper function to convert region state list to tensor for the model."""
        return torch.tensor(region_state, dtype=torch.float32)

    def store_transition(self,
                         state: List[float],
                         action: int,
                         reward: float,
                         next_state: List[float],
                         done: bool,
                         actor_type: str,
                         region_local_state,
                         next_region_local_state,
                         actor_index: int):
        self.memory.push(
            state, action, reward, next_state, done, actor_type,
            region_local_state, next_region_local_state, actor_index
        )

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)

        self.optimizer.zero_grad()
        total_loss = 0.0

        for transition in batch:
            current_state_tensor = self._build_augmented_state(
                single_state=transition.state,
                region_local_state=transition.region_local_state,
                actor_index=transition.actor_index,
                use_target=False
            )
            current_q_values = self.policy_net(current_state_tensor)
            chosen_q = current_q_values[0, transition.action]

            with torch.no_grad():
                next_state_policy_tensor = self._build_augmented_state(
                    single_state=transition.next_state,
                    region_local_state=transition.next_region_local_state,
                    actor_index=transition.actor_index,
                    use_target=False
                )
                next_q_values_policy = self.policy_net(next_state_policy_tensor)[0]

                mask = torch.full(next_q_values_policy.shape, float('-inf'), device=device)
                valid_actions = VALID_ACTIONS[transition.actor_type]
                mask[valid_actions] = 0.0
                best_action = torch.argmax(next_q_values_policy + mask).item()

                next_state_target_tensor = self._build_augmented_state(
                    single_state=transition.next_state,
                    region_local_state=transition.next_region_local_state,
                    actor_index=transition.actor_index,
                    use_target=True
                )
                next_q_values_target = self.target_net(next_state_target_tensor)[0]
                max_next_q = next_q_values_target[best_action]

                target_q = transition.reward + self.gamma * max_next_q * (1 - float(transition.done))

            total_loss = total_loss + nn.MSELoss()(chosen_q, target_q)

        loss = total_loss / len(batch)

        loss.backward()
        self.optimizer.step()

        self.update_step += 1
        if self.update_step % self.target_update == 0:
            self.target_gnn.load_state_dict(self.policy_gnn.state_dict())
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def update_epsilon(self):
        """
        Decays epsilon after each episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
