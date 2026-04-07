import numpy as np
import traci
import typing
import tls_states
from tls_states import GREEN_DURATION, ACTION_LENGTHS
from tls_states import LOGICS
from typing import List, Dict
from Region import Actor, RegionController
from Region import N_ACTION

"""
1- total number of vehicles - halted vehicles (on incoming edges) _onehop_approaching_flow
2- number of halted vehicles in each side of its one-hop neighbors _onehop_neighbors_context
3- one-hop neighbors current action _onehop_neighbors_context
4- remaining time of their chosen action _onehop_neighbors_context
"""


class Environment:
    WINDOW = tls_states.GREEN_DURATION + tls_states.YELLOW_DURATION

    def __init__(self, sumo_cfg: str, regions: Dict[int, List[Actor]], region_controllers: List[RegionController],
                _same_region_onehop_neighbors:Dict[str,List]):
        self.sumo_cfg = sumo_cfg  # config file
        self.regions = regions  # values are Actor objects containing ID and type and tls_state
        self.max_step = 18000  # total simulation time in seconds
        self.regions_states = {}
        self.region_controllers = region_controllers

        self.same_region_onehop_neighbors = _same_region_onehop_neighbors

        self.junction_to_region = {}
        for region_id, actors in self.regions.items():
            for actor in actors:
                self.junction_to_region[actor.name] = region_id

        self.NEIGHBOR_REWARD_IMPACT = 0.3


    def load_tls_program_logics(self) -> None:
        """
        This method loads all the predefined ProgramLogics in the coresponding
        tls object.
        Two main types of actors -> 3ways and 4ways
        each action has 3 predefined tls logics.
        """
        all_tls_IDs = traci.trafficlight.getIDList()  # a tuple of strings containing tls IDS
        for tls_ID in all_tls_IDs:
            if tls_ID[0] == "A":
                # ACTION 0
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_S_3A_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_S_3A_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_S_3A_EXTEND"])
                # ACTION 1
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_L_3A_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_L_3A_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_L_3A_EXTEND"])
                # ACTION 3
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["L_3A_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["L_3A_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["L_3A_EXTEND"])

            elif tls_ID[0] == "E":
                # ACTION 0
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_S_3E_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_S_3E_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_S_3E_EXTEND"])
                # ACTION 1
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_L_3E_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_L_3E_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_L_3E_EXTEND"])
                # ACTION 3
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["L_3E_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["L_3E_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["L_3E_EXTEND"])

            elif tls_ID[1] == "0":
                # ACTION 1
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["L_30_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["L_30_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["L_30_EXTEND"])
                # ACTION 2
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_S_30_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_S_30_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_S_30_EXTEND"])
                # ACTION 3
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_L_30_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_L_30_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_L_30_EXTEND"])

            elif tls_ID[1] == "4":
                # ACTION 1
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["L_34_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["L_34_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["L_34_EXTEND"])
                # ACTION 2
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_S_34_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_S_34_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_S_34_EXTEND"])
                # ACTION 3
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_L_34_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_L_34_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_L_34_EXTEND"])


            else:  # 9 4way actors
                # ACTION 0
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_S_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_S_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_S_EXTEND"])
                # ACTION 1
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_L_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_L_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["NS_L_EXTEND"])
                # ACTION 2
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_S_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_S_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_S_EXTEND"])
                # ACTION 3
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_L_GREEN"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_L_YELLOW"])
                traci.trafficlight.setProgramLogic(tls_ID, LOGICS["EW_L_EXTEND"])

    def start(self, agents):
        traci.start(["sumo", "-c", self.sumo_cfg])
        self.load_tls_program_logics()
        self.initialize_logics(agents)

    def reset(self, agents):
        traci.load(["-c", self.sumo_cfg])
        self.regions_states = {}
        self.load_tls_program_logics()
        self.initialize_logics(agents)
        # reset train step for each RegionController
        for region in self.region_controllers:
            region: RegionController
            region.update_step = 0

    def initialize_logics(self, agents):
        for num, region_controller in agents.items():
            for actor in region_controller.actors:
                actor: Actor
                if actor.type == "3_A":
                    traci.trafficlight.setProgram(tlsID=actor.name, programID="NS_S_3A_GREEN")
                    actor.current_action = 0
                elif actor.type == "3_E":
                    traci.trafficlight.setProgram(tlsID=actor.name, programID="NS_S_3E_GREEN")
                    actor.current_action = 0
                elif actor.type == "3_0":
                    traci.trafficlight.setProgram(tlsID=actor.name, programID="L_30_GREEN")
                    actor.current_action = 1
                elif actor.type == "3_4":
                    traci.trafficlight.setProgram(tlsID=actor.name, programID="L_34_GREEN")
                    actor.current_action = 1
                else:
                    traci.trafficlight.setProgram(tlsID=actor.name, programID="NS_S_GREEN")
                    actor.current_action = 0
                actor.state = self.get_agent_state(actor.name, 0)
                actor.next_time = GREEN_DURATION
                actor.is_yellow = False

    def change_logic(self, actor_name: str, actor_type: str, chosen_action: int) -> None:
        """
        Performs **Green** action for a given actor. (only actions of *_GREEN type)
        """
        if actor_type == "3_A":
            if chosen_action == 0:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_S_3A_GREEN")
            elif chosen_action == 1:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_L_3A_GREEN")
            elif chosen_action == 3:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="L_3A_GREEN")
        elif actor_type == "3_E":
            if chosen_action == 0:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_S_3E_GREEN")
            elif chosen_action == 1:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_L_3E_GREEN")
            elif chosen_action == 3:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="L_3E_GREEN")
        elif actor_type == "3_0":
            if chosen_action == 1:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="L_30_GREEN")
            elif chosen_action == 2:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_S_30_GREEN")
            elif chosen_action == 3:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_L_30_GREEN")
        elif actor_type == "3_4":
            if chosen_action == 1:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="L_34_GREEN")
            elif chosen_action == 2:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_S_34_GREEN")
            elif chosen_action == 3:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_L_34_GREEN")
        else:  # 4
            if chosen_action == 0:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_S_GREEN")
            elif chosen_action == 1:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_L_GREEN")
            elif chosen_action == 2:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_S_GREEN")
            elif chosen_action == 3:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_L_GREEN")

    def set_yellow_logic(self, actor_name: str, actor_type: str, current_action: int) -> None:
        """
        Performs **YELLOW** action for a given actor. (only actions of *_YELLOW type)
        """
        if actor_type == "3_A":
            if current_action == 0 or current_action == 4:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_S_3A_YELLOW")
            elif current_action == 1 or current_action == 5:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_L_3A_YELLOW")
            elif current_action == 3 or current_action == 7:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="L_3A_YELLOW")
        elif actor_type == "3_E":
            if current_action == 0 or current_action == 4:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_S_3E_YELLOW")
            elif current_action == 1 or current_action == 5:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_L_3E_YELLOW")
            elif current_action == 3 or current_action == 7:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="L_3E_YELLOW")
        elif actor_type == "3_0":
            if current_action == 1 or current_action == 5:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="L_30_YELLOW")
            elif current_action == 2 or current_action == 6:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_S_30_YELLOW")
            elif current_action == 3 or current_action == 7:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_L_30_YELLOW")
        elif actor_type == "3_4":
            if current_action == 1 or current_action == 5:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="L_34_YELLOW")
            elif current_action == 2 or current_action == 6:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_S_34_YELLOW")
            elif current_action == 3 or current_action == 7:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_L_34_YELLOW")
        else:  # 4
            if current_action == 0 or current_action == 4:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_S_YELLOW")
            elif current_action == 1 or current_action == 5:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_L_YELLOW")
            elif current_action == 2 or current_action == 6:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_S_YELLOW")
            elif current_action == 3 or current_action == 7:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_L_YELLOW")

    def extend_logic(self, actor_name: str, actor_type: str, current_action: int) -> None:
        """
        Performs **EXTEND** action for a given actor. (only actions of *_EXTEND type)
        """
        if actor_type == "3_A":
            if current_action == 4:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_S_3A_EXTEND")
            elif current_action == 5:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_L_3A_EXTEND")
            elif current_action == 7:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="L_3A_EXTEND")
        elif actor_type == "3_E":
            if current_action == 4:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_S_3E_EXTEND")
            elif current_action == 5:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_L_3E_EXTEND")
            elif current_action == 7:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="L_3E_EXTEND")
        elif actor_type == "3_0":
            if current_action == 5:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="L_30_EXTEND")
            elif current_action == 6:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_S_30_EXTEND")
            elif current_action == 7:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_L_30_EXTEND")
        elif actor_type == "3_4":
            if current_action == 5:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="L_34_EXTEND")
            elif current_action == 6:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_S_34_EXTEND")
            elif current_action == 7:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_L_34_EXTEND")
        else:  # 4
            if current_action == 4:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_S_EXTEND")
            elif current_action == 5:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="NS_L_EXTEND")
            elif current_action == 6:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_S_EXTEND")
            elif current_action == 7:
                traci.trafficlight.setProgram(tlsID=actor_name, programID="EW_L_EXTEND")

    def _action_one_hot(self, action):
        arr = np.zeros(N_ACTION)
        arr[action] = 1
        return list(arr)

    def step(self, region_actions: Dict[int, List[int]], current_step: int) -> tuple[
        Dict[int, List[List]], Dict[int, List], bool]:
        """
            region_actions:
                - Determines all the actions to be taken by all the actors in that region.
                - The order matches the order that these actors/junctions were passed to
                  RegionController __init__.

            ACTION 0 : NS_S
            ACTION 1 : NS_L
            ACTION 2 : EW_S
            ACTION 3 : EW_L
            These mapping are also documented in tls_states.py
        """

        for region_id, actions in region_actions.items():  # PERFORMS ACTION FOR EACH JUNCTION IN EACH REGION
            region_actors = self.regions[region_id]
            for i, action in enumerate(actions):
                actor: Actor
                actor = region_actors[i]
                if actor.type == "3_A":
                    if action == 0:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["NS_S_3A"])
                    elif action == 1:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["NS_L_3A"])
                    elif action == 3:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["L_3A"])
                elif actor.type == "3_E":
                    if action == 0:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["NS_S_3E"])
                    elif action == 1:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["NS_L_3E"])
                    elif action == 3:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["L_3E"])
                elif actor.type == "3_0":
                    if action == 1:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["L_30"])
                    elif action == 2:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["EW_S_30"])
                    elif action == 3:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["EW_L_30"])
                elif actor.type == "3_4":
                    if action == 1:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["L_34"])
                    elif action == 2:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["EW_S_34"])
                    elif action == 3:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["EW_L_34"])
                else:  # 4
                    if action == 0:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["NS_S"])
                    elif action == 1:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["NS_L"])
                    elif action == 2:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["EW_S"])
                    elif action == 3:
                        traci.trafficlight.setProgramLogic(tlsID=actor.name, logic=LOGICS["EW_L"])

                        # TODO check if we have simulated window seconds, also update current_step in MAIN
        if current_step + self.WINDOW > self.max_step:
            simulate_to = self.max_step
        else:
            simulate_to = current_step + self.WINDOW
        traci.simulationStep(step=float(simulate_to))

        next_states = {}
        rewards = {}

        for region_id, actions in region_actions.items():  # COLLECTS REWARDS AND RETREIVE NEXT STATES
            region_rewards = []
            next_region_states = []
            region_actors = self.regions[region_id]
            for i, action in enumerate(actions):
                actor = region_actors[i]
                junction_state = self.get_agent_state(actor.name)
                junction_reward = self.get_actor_reward(actor.name)

                next_region_states.append(junction_state)
                region_rewards.append(junction_reward)

            next_states[region_id] = next_region_states
            rewards[region_id] = region_rewards

        self.regions_states = next_states
        if simulate_to < self.max_step:
            done = False
        else:
            done = True

        return next_states, rewards, done

    def get_agent_state(self, junction_id: str, action: int) -> List:
        """
        one hot enncoding for 5 junction types
        [1,0,0,0,0] 3_A
        [0,1,0,0,0] 3_E
        [0,0,1,0,0] 3_0
        [0,0,0,1,0] 3_4
        [0,0,0,0,1] 4
        :param junction_id:
        :return: a list featuring [NV_v, NV_h,left_lane_NV_v, left_lane_NV_h, *,*,*,*,*]
        """
        #TODO use self._vh_halts here too!
        edge = [x for x in traci.junction.getIncomingEdges(junction_id) if "_" not in x]
        if junction_id[0] in ['A', 'E']:
            type = [1, 0, 0, 0, 0] if junction_id[0] == "A" else [0, 1, 0, 0, 0]
            if junction_id[0] == "A":
                edge_vertical = [edge[0], edge[1]]
                edge_horizontal = [edge[2]]
                left_lane_vertical = [f"{edge_vertical[1]}_2"]
                left_lane_horizontal = [f"{edge_horizontal[0]}_2"]
            else:
                edge_vertical = [edge[1], edge[2]]
                edge_horizontal = [edge[0]]
                left_lane_vertical = [f"{edge_vertical[0]}_2"]
                left_lane_horizontal = [f"{edge_horizontal[0]}_2"]

            vertical_num_vehicle = (traci.edge.getLastStepHaltingNumber(
                edge_vertical[0]) + traci.edge.getLastStepHaltingNumber(edge_vertical[1])) / 2
            horizontal_num_vehicle = traci.edge.getLastStepHaltingNumber(edge_horizontal[0])

            vertical_left_lane_num_vehicle = traci.lane.getLastStepHaltingNumber(left_lane_vertical[0])
            horizontal_left_lane_num_vehicle = traci.lane.getLastStepHaltingNumber(left_lane_horizontal[0]) 

        elif junction_id[1] in ['0', '4']:
            type = [0, 0, 1, 0, 0] if junction_id[1] == "0" else [0, 0, 0, 1, 0]
            edge_vertical = [edge[1]]
            edge_horizontal = [edge[0], edge[2]]

            if junction_id[1] == "0":
                left_lane_vertical = [f"{edge_vertical[0]}_2"]
                left_lane_horizontal = [f"{edge_horizontal[0]}_2"]
            else:
                left_lane_vertical = [f"{edge_vertical[0]}_2"]
                left_lane_horizontal = [f"{edge_horizontal[1]}_2"]

                
            vertical_num_vehicle = traci.edge.getLastStepHaltingNumber(edge_vertical[0])
            horizontal_num_vehicle = (traci.edge.getLastStepHaltingNumber(
                edge_horizontal[0]) + traci.edge.getLastStepHaltingNumber(edge_horizontal[1])) / 2

            vertical_left_lane_num_vehicle = traci.lane.getLastStepHaltingNumber(left_lane_vertical[0])
            horizontal_left_lane_num_vehicle = traci.lane.getLastStepHaltingNumber(left_lane_horizontal[0]) 


        else:  # order of returned edges is [LEFT, DOWN, UP, RIGHT]
            type = [0, 0, 0, 0, 1]
            edge_vertical = [edge[1], edge[2]]
            edge_horizontal = [edge[0], edge[3]]

            left_lane_vertical = [f"{edge_vertical[0]}_2",f"{edge_vertical[1]}_2"]
            left_lane_horizontal = [f"{edge_horizontal[0]}_2",f"{edge_horizontal[1]}_2"]

            vertical_num_vehicle = (traci.edge.getLastStepHaltingNumber(
                edge_vertical[0]) + traci.edge.getLastStepHaltingNumber(edge_vertical[1])) / 2
            horizontal_num_vehicle = (traci.edge.getLastStepHaltingNumber(
                edge_horizontal[0]) + traci.edge.getLastStepHaltingNumber(edge_horizontal[1])) / 2

            vertical_left_lane_num_vehicle = (traci.lane.getLastStepHaltingNumber(
                left_lane_vertical[0]) + traci.lane.getLastStepHaltingNumber(left_lane_vertical[1])) / 2

            horizontal_left_lane_num_vehicle = (traci.lane.getLastStepHaltingNumber(
                left_lane_horizontal[0]) + traci.lane.getLastStepHaltingNumber(left_lane_horizontal[1])) / 2


        state = [vertical_num_vehicle, horizontal_num_vehicle, vertical_left_lane_num_vehicle, horizontal_left_lane_num_vehicle]
        state.extend(type)
        state.extend(self._action_one_hot(action))
        
        # one-hop context
        onehop_context = self._onehop_context(junction_id)
        state.extend(onehop_context)
        return state


    def get_actor_reward_onehop_centralized(self, junction_id: str) -> float:
        """
        Calculates the reward for a given junction in a centralized one-hop environment.
        The reward is computed as the negative sum of the local queue length (normalized by the number of edges)
        and a weighted sum of the average queue lengths of the one-hop neighboring junctions in the same region.  

        see self.NEIGHBOR_REWARD_IMPACT  
        """
        halts,num_edges = self._actor_queue_len(junction_id)
        r_loc = halts/num_edges
        r_onehop = 0 
        for onehop_neighbor_id in self.same_region_onehop_neighbors[junction_id]:
            if not onehop_neighbor_id:
                continue
            else:
                onehop_neighbor_halts, num_edges = self._actor_queue_len(onehop_neighbor_id)
                r_onehop += onehop_neighbor_halts / num_edges  # average queue length of the neighbor
        return -(r_loc + self.NEIGHBOR_REWARD_IMPACT * r_onehop)


    def _actor_queue_len(self, junction_id: str) -> tuple[int,int]:
        """Sum of halting vehicles on all incoming edges of a junction and the num of incoming edges."""
        edges = [x for x in traci.junction.getIncomingEdges(junction_id) if "_" not in x]
        queue_length = 0
        for edge in edges:
            queue_length += traci.edge.getLastStepHaltingNumber(edge)
        return int(queue_length), len(edges)
    
    def get_region_reward(self, region_id) -> int:
        """
        Retruns sum of final rewards of all actors in a region.
        """
        reward = 0
        for actor in self.regions[region_id]:
            reward += self.get_actor_reward_onehop_centralized(actor.name)
        return reward

    def close(self):
        traci.close()

    def _vh_halts(self, junction_id:str) -> tuple[float, float,float, float]:
        """
        Returns vertical and horizontal halts and horizontal and vertical left lanes halts
        of a junction divided by 2 in directions with two incoming edges.
        """
        edge = [x for x in traci.junction.getIncomingEdges(junction_id) if "_" not in x]
        if junction_id[0] in ['A', 'E']:
            type = [1, 0, 0, 0, 0] if junction_id[0] == "A" else [0, 1, 0, 0, 0]
            if junction_id[0] == "A":
                edge_vertical = [edge[0], edge[1]]
                edge_horizontal = [edge[2]]
                left_lane_vertical = [f"{edge_vertical[1]}_2"]
                left_lane_horizontal = [f"{edge_horizontal[0]}_2"]
            else:
                edge_vertical = [edge[1], edge[2]]
                edge_horizontal = [edge[0]]
                left_lane_vertical = [f"{edge_vertical[0]}_2"]
                left_lane_horizontal = [f"{edge_horizontal[0]}_2"]

            vertical_num_vehicle = (traci.edge.getLastStepHaltingNumber(
                edge_vertical[0]) + traci.edge.getLastStepHaltingNumber(edge_vertical[1])) / 2
            horizontal_num_vehicle = traci.edge.getLastStepHaltingNumber(edge_horizontal[0])

            vertical_left_lane_num_vehicle = traci.lane.getLastStepHaltingNumber(left_lane_vertical[0])
            horizontal_left_lane_num_vehicle = traci.lane.getLastStepHaltingNumber(left_lane_horizontal[0]) 

        elif junction_id[1] in ['0', '4']:
            type = [0, 0, 1, 0, 0] if junction_id[0] == "0" else [0, 0, 0, 1, 0]
            edge_vertical = [edge[1]]
            edge_horizontal = [edge[0], edge[2]]

            if junction_id[1] == "0":
                left_lane_vertical = [f"{edge_vertical[0]}_2"]
                left_lane_horizontal = [f"{edge_horizontal[0]}_2"]
            else:
                left_lane_vertical = [f"{edge_vertical[0]}_2"]
                left_lane_horizontal = [f"{edge_horizontal[1]}_2"]

                
            vertical_num_vehicle = traci.edge.getLastStepHaltingNumber(edge_vertical[0])
            horizontal_num_vehicle = (traci.edge.getLastStepHaltingNumber(
                edge_horizontal[0]) + traci.edge.getLastStepHaltingNumber(edge_horizontal[1])) / 2

            vertical_left_lane_num_vehicle = traci.lane.getLastStepHaltingNumber(left_lane_vertical[0])
            horizontal_left_lane_num_vehicle = traci.lane.getLastStepHaltingNumber(left_lane_horizontal[0]) 


        else:  # order of returned edges is [LEFT, DOWN, UP, RIGHT]
            type = [0, 0, 0, 0, 1]
            edge_vertical = [edge[1], edge[2]]
            edge_horizontal = [edge[0], edge[3]]

            left_lane_vertical = [f"{edge_vertical[0]}_2",f"{edge_vertical[1]}_2"]
            left_lane_horizontal = [f"{edge_horizontal[0]}_2",f"{edge_horizontal[1]}_2"]

            vertical_num_vehicle = (traci.edge.getLastStepHaltingNumber(
                edge_vertical[0]) + traci.edge.getLastStepHaltingNumber(edge_vertical[1])) / 2
            horizontal_num_vehicle = (traci.edge.getLastStepHaltingNumber(
                edge_horizontal[0]) + traci.edge.getLastStepHaltingNumber(edge_horizontal[1])) / 2

            vertical_left_lane_num_vehicle = (traci.lane.getLastStepHaltingNumber(
                left_lane_vertical[0]) + traci.lane.getLastStepHaltingNumber(left_lane_vertical[1])) / 2

            horizontal_left_lane_num_vehicle = (traci.lane.getLastStepHaltingNumber(
                left_lane_horizontal[0]) + traci.lane.getLastStepHaltingNumber(left_lane_horizontal[1])) / 2

        
        return vertical_num_vehicle, horizontal_num_vehicle, vertical_left_lane_num_vehicle,horizontal_left_lane_num_vehicle


    def _onehop_context(self, junction_id: int) -> list:
        vert_onehop_approaching_flow, hor_onehop_approaching_flow = self._onehop_approaching_flow(junction_id)
        n_onehop_neighbor_context,e_onehop_neighbor_context,s_onehop_neighbor_context,w_onehop_neighbor_context=self._onehop_neighbors_context(junction_id)
        onehop_context = [vert_onehop_approaching_flow,hor_onehop_approaching_flow]
        onehop_context.extend(
                        n_onehop_neighbor_context + e_onehop_neighbor_context + 
                        s_onehop_neighbor_context + w_onehop_neighbor_context)
        return onehop_context
    
    def _onehop_approaching_flow(self,junction_id: int) -> tuple[int, int]:
        edge = [x for x in traci.junction.getIncomingEdges(junction_id) if "_" not in x]
        if junction_id[0] in ['A', 'E']:
            if junction_id[0] == "A":
                edge_vertical = [edge[0], edge[1]]
                edge_horizontal = [edge[2]]
            else:
                edge_vertical = [edge[1], edge[2]]
                edge_horizontal = [edge[0]]

            vertical_approaching_flow = (traci.edge.getLastStepVehicleNumber(edge_vertical[0]) - traci.edge.getLastStepHaltingNumber(edge_vertical[0])) \
                                        + (traci.edge.getLastStepVehicleNumber(edge_vertical[1]) - traci.edge.getLastStepHaltingNumber(edge_vertical[1]))
            horizontal_approaching_flow = (traci.edge.getLastStepVehicleNumber(edge_horizontal[0]) - traci.edge.getLastStepHaltingNumber(edge_horizontal[0]))

        elif junction_id[1] in ['0', '4']:
            edge_vertical = [edge[1]]
            edge_horizontal = [edge[0], edge[2]]

            vertical_approaching_flow = (traci.edge.getLastStepVehicleNumber(edge_vertical[0]) - traci.edge.getLastStepHaltingNumber(edge_vertical[0])) 
            horizontal_approaching_flow = (traci.edge.getLastStepVehicleNumber(edge_horizontal[0]) - traci.edge.getLastStepHaltingNumber(edge_horizontal[0])) \
                                    + (traci.edge.getLastStepVehicleNumber(edge_horizontal[1]) - traci.edge.getLastStepHaltingNumber(edge_horizontal[1]))


        else:  # order of returned edges is [LEFT, DOWN, UP, RIGHT]
            edge_vertical = [edge[1], edge[2]]
            edge_horizontal = [edge[0], edge[3]]
        
            vertical_approaching_flow = (traci.edge.getLastStepVehicleNumber(edge_vertical[0]) - traci.edge.getLastStepHaltingNumber(edge_vertical[0])) \
                                        + (traci.edge.getLastStepVehicleNumber(edge_vertical[1]) - traci.edge.getLastStepHaltingNumber(edge_vertical[1]))
            horizontal_approaching_flow = (traci.edge.getLastStepVehicleNumber(edge_horizontal[0]) - traci.edge.getLastStepHaltingNumber(edge_horizontal[0])) \
                                        + (traci.edge.getLastStepVehicleNumber(edge_horizontal[1]) - traci.edge.getLastStepHaltingNumber(edge_horizontal[1]))

        return vertical_approaching_flow, horizontal_approaching_flow

    
    def _onehop_neighbors_context(self,junction_id: int) -> tuple[list,list,list,list]: 
        """
        Returns 4*13 vector containing onehop same region neighbors.
        
        Each 13-d vector of a neighbor is as below:

        - v_halts, h_halts, v_left_halts, h_left_halts (4)
        - one-hot current action (N_ACTION)
        - remaining seconds until next action decision (1)

        """
        t = traci.simulation.getTime()

        onehop_neighbors_context = ([],[],[],[])
        for i,onehop_neighbor_id in enumerate(self.same_region_onehop_neighbors[junction_id]):
            if not onehop_neighbor_id:
                onehop_neighbors_context[i].extend(np.zeros(4+N_ACTION+1).tolist())  # only intra region observations
                continue
            onehop_neighbor = None # if instanciated, is of type Actor
            for actor in self.regions[self.junction_to_region[junction_id]]: # find Actor obj of the neighbor
                if actor.name == onehop_neighbor_id:
                    onehop_neighbor = actor
                    break
            if not onehop_neighbor:
                import ipdb; ipdb.set_trace()
                raise RuntimeError("onehop neighbor was not found!")
            (onehop_neighbor_v_halts, onehop_neighbor_h_halts ,
            onehop_neighbor_v_left_halts, onehop_neighbor_h_left_halts) =self._vh_halts(onehop_neighbor.name)
            remaining_seconds = onehop_neighbor.next_time - t
            if onehop_neighbor.is_yellow:
                 # remaining_seconds has stored seconds to the end of yellow phase till now
                 remaining_seconds += ACTION_LENGTHS[onehop_neighbor.current_action]

            onehop_neighbors_context[i].extend([onehop_neighbor_v_halts,onehop_neighbor_h_halts,
                                        onehop_neighbor_v_left_halts, onehop_neighbor_h_left_halts] + \
                                        self._action_one_hot(onehop_neighbor.current_action) + \
                                         [remaining_seconds])
        return onehop_neighbors_context


