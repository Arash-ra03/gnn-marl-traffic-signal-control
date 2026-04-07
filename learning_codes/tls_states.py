import traci



num_logics = 4
YELLOW_DURATION = 3
GREEN_DURATION = 15
EXTEND_DURATION = 5

# not necessarily extend meaning
# 0 NS_S
# 1 NS_L
# 2 EW_S
# 3 EW_L
# 4 NS_S EXTEND
# 5 NS_L EXTEND
# 6 EW_S EXTEND
# 7 EW_L EXTEND

VALID_ACTIONS = {
    "3_A": [0,1,3,4,5,7],
    "3_E":[0,1,3,4,5,7],
    "3_4":[1,2,3,5,6,7,],
    "3_0":[1,2,3,5,6,7],
    "4":[0,1,2,3,4,5,6,7],
}

ACTION_LENGTHS = {
    0:GREEN_DURATION,
    1:GREEN_DURATION,
    2:GREEN_DURATION,
    3:GREEN_DURATION,
    4:EXTEND_DURATION,
    5:EXTEND_DURATION,
    6:EXTEND_DURATION,
    7:EXTEND_DURATION,
}

SOFT_TRANSITION = {
    0: [0, 4],
    1: [1, 5],
    2: [2, 6],
    3: [3, 7],
    4: [0, 4],
    5: [1, 5],
    6: [2, 6],
    7: [3, 7],
}
phase_NS_S = {"green": "GGGrrrrrGGGrrrrr", "yellow": "GyyrrrrrGyyrrrrr"}
phase_NS_L = {"green": "GrrGrrrrGrrGrrrr", "yellow": "yrryrrrryrryrrrr"}
phase_EW_S = {"green": "rrrrGGGrrrrrGGGr", "yellow": "rrrrGyyrrrrrGyyr"}
phase_EW_L = {"green": "rrrrGrrGrrrrGrrG", "yellow": "rrrryrryrrrryrry"}
#####################
phase_NS_S_3E = {"green":"GGGGGGGrrrrr", "yellow": "Gyyyyyyrrrrr"}
phase_NS_L_3E = {"green":"GrrrrrrGGGrr", "yellow": "GrrrrrryyGrr"}
phase_L_3E = {"green":"GrrrrrrrrGGG", "yellow": "Grrrrrrrryyy"} # equivalant to EW_L
#####################
phase_NS_S_3A = {"green": "GGGrrrrrGGGG", "yellow": "yyyrrrrrGyyy"}
phase_NS_L_3A = {"green": "rrrGGGrrGrrr", "yellow": "rrryyGrrGrrr"}
phase_L_3A = {"green": "rrrrrGGGGrrr", "yellow": "rrrrryyyGrrr"} # equivalant to EW_L
#####################
phase_EW_S_34 = {"green": "GGGrrrrrGGGG", "yellow": "yyyrrrrrGyyy"}
phase_EW_L_34 = {"green": "rrrGGGrrGrrr", "yellow": "rrryyGrrGrrr"}
phase_L_34 = {"green": "rrrrrGGGGrrr", "yellow": "rrrrryyyGrrr"} # equivalant to NS_L
##################### 
phase_EW_S_30 = {"green": "rrrGGGGGGGrr", "yellow": "rrrGyyyyyyrr"}
phase_EW_L_30 = {"green": "GrrGrrrrrrGG", "yellow": "GrrGrrrrrryy"}
phase_L_30 = {"green": "GGGGrrrrrrrr", "yellow": "yyyGrrrrrrrr"} # equivalant to NS_L


#NOTE 3 Logics for each action
# - Green
# - Yellow, not a choosable action (transition between actions)
# - Extend Green (same as the first one with different duration)
LOGICS = {}
phases = [] # ACTION 0
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_NS_S['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_NS_S['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_NS_S['green']))
LOGICS["NS_S_GREEN"] = traci.trafficlight.Logic("NS_S_GREEN",0, 0, [phases[0]])
LOGICS["NS_S_YELLOW"] = traci.trafficlight.Logic("NS_S_YELLOW", 0, 0, [phases[1]])
# ACTION 4
LOGICS["NS_S_EXTEND"] = traci.trafficlight.Logic("NS_S_EXTEND",0, 0, [phases[2]])
##############################################################
phases = [] # ACTION 1
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_NS_L['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_NS_L['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_NS_L['green']))
LOGICS["NS_L_GREEN"] = traci.trafficlight.Logic("NS_L_GREEN",0, 0, [phases[0]])
LOGICS["NS_L_YELLOW"] = traci.trafficlight.Logic("NS_L_YELLOW", 0, 0, [phases[1]])
LOGICS["NS_L_EXTEND"] = traci.trafficlight.Logic("NS_L_EXTEND",0, 0, [phases[2]])
##############################################################
phases = [] # ACTION 2
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_EW_S['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_EW_S['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_EW_S['green']))
LOGICS["EW_S_GREEN"] = traci.trafficlight.Logic("EW_S_GREEN",0, 0, [phases[0]])
LOGICS["EW_S_YELLOW"] = traci.trafficlight.Logic("EW_S_YELLOW", 0, 0, [phases[1]])
# ACTION 6
LOGICS["EW_S_EXTEND"] = traci.trafficlight.Logic("EW_S_EXTEND",0, 0, [phases[2]])
##############################################################
phases = [] # ACTION 3  
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_EW_L['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_EW_L['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_EW_L['green']))
LOGICS["EW_L_GREEN"] = traci.trafficlight.Logic("EW_L_GREEN",0, 0, [phases[0]])
LOGICS["EW_L_YELLOW"] = traci.trafficlight.Logic("EW_L_YELLOW", 0, 0, [phases[1]])
LOGICS["EW_L_EXTEND"] = traci.trafficlight.Logic("EW_L_EXTEND",0, 0, [phases[2]])
##############################################################
##############################################################
##############################################################

phases = [] # ACTION 0
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_NS_S_3E['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_NS_S_3E['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_NS_S_3E['green']))
LOGICS["NS_S_3E_GREEN"] = traci.trafficlight.Logic("NS_S_3E_GREEN",0, 0, [phases[0]])
LOGICS["NS_S_3E_YELLOW"] = traci.trafficlight.Logic("NS_S_3E_YELLOW", 0, 0, [phases[1]])
# ACTION 4
LOGICS["NS_S_3E_EXTEND"] = traci.trafficlight.Logic("NS_S_3E_EXTEND",0, 0, [phases[2]])
##############################################################
phases = [] # ACTION 1
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_NS_L_3E['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_NS_L_3E['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_NS_L_3E['green']))
LOGICS["NS_L_3E_GREEN"] = traci.trafficlight.Logic("NS_L_3E_GREEN",0, 0, [phases[0]])
LOGICS["NS_L_3E_YELLOW"] = traci.trafficlight.Logic("NS_L_3E_YELLOW", 0, 0, [phases[1]])
# ACTION 5
LOGICS["NS_L_3E_EXTEND"] = traci.trafficlight.Logic("NS_L_3E_EXTEND",0, 0, [phases[2]])
##############################################################
phases = [] # ACTION 3
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_L_3E['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_L_3E['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_L_3E['green']))
LOGICS["L_3E_GREEN"] = traci.trafficlight.Logic("L_3E_GREEN",0, 0, [phases[0]])
LOGICS["L_3E_YELLOW"] = traci.trafficlight.Logic("L_3E_YELLOW", 0, 0, [phases[1]])
# ACTION 7
LOGICS["L_3E_EXTEND"] = traci.trafficlight.Logic("L_3E_EXTEND",0, 0, [phases[2]])
##############################################################
##############################################################
##############################################################

phases = [] # ACTION 0
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_NS_S_3A['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_NS_S_3A['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_NS_S_3A['green']))
LOGICS["NS_S_3A_GREEN"] = traci.trafficlight.Logic("NS_S_3A_GREEN",0, 0, [phases[0]])
LOGICS["NS_S_3A_YELLOW"] = traci.trafficlight.Logic("NS_S_3A_YELLOW", 0, 0,  [phases[1]])
# ACTION 4
LOGICS["NS_S_3A_EXTEND"] = traci.trafficlight.Logic("NS_S_3A_EXTEND", 0, 0,  [phases[2]])
##############################################################
phases = [] # ACTION 1
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_NS_L_3A['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_NS_L_3A['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_NS_L_3A['green']))
LOGICS["NS_L_3A_GREEN"] = traci.trafficlight.Logic("NS_L_3A_GREEN",0, 0, [phases[0]])
LOGICS["NS_L_3A_YELLOW"] = traci.trafficlight.Logic("NS_L_3A_YELLOW", 0, 0, [phases[1]])
# ACTION 5
LOGICS["NS_L_3A_EXTEND"] = traci.trafficlight.Logic("NS_L_3A_EXTEND", 0, 0, [phases[2]])
##############################################################
phases = [] # ACTION 3
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_L_3A['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_L_3A['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_L_3A['green']))
LOGICS["L_3A_GREEN"] = traci.trafficlight.Logic("L_3A_GREEN",0, 0, [phases[0]])
LOGICS["L_3A_YELLOW"] = traci.trafficlight.Logic("L_3A_YELLOW", 0, 0, [phases[1]])
# ACTION 7
LOGICS["L_3A_EXTEND"] = traci.trafficlight.Logic("L_3A_EXTEND", 0, 0, [phases[2]])
##############################################################
##############################################################
##############################################################

phases = [] # ACTION 2
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_EW_S_34['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_EW_S_34['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_EW_S_34['green']))
LOGICS["EW_S_34_GREEN"] = traci.trafficlight.Logic("EW_S_34_GREEN",0, 0, [phases[0]])
LOGICS["EW_S_34_YELLOW"] = traci.trafficlight.Logic("EW_S_34_YELLOW", 0, 0,  [phases[1]])
# ACTION 6
LOGICS["EW_S_34_EXTEND"] = traci.trafficlight.Logic("EW_S_34_EXTEND", 0, 0,  [phases[2]])
##############################################################
phases = [] # ACTION 3
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_EW_L_34['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_EW_L_34['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_EW_L_34['green']))
LOGICS["EW_L_34_GREEN"] = traci.trafficlight.Logic("EW_L_34_GREEN",0, 0, [phases[0]])
LOGICS["EW_L_34_YELLOW"] = traci.trafficlight.Logic("EW_L_34_YELLOW", 0, 0, [phases[1]])
# ACTION 7
LOGICS["EW_L_34_EXTEND"] = traci.trafficlight.Logic("EW_L_34_EXTEND", 0, 0, [phases[2]])
##############################################################
phases = [] # ACTION 1
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_L_34['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_L_34['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_L_34['green']))
LOGICS["L_34_GREEN"] = traci.trafficlight.Logic("L_34_GREEN",0, 0, [phases[0]])
LOGICS["L_34_YELLOW"] = traci.trafficlight.Logic("L_34_YELLOW", 0, 0, [phases[1]])
# ACTION 5
LOGICS["L_34_EXTEND"] = traci.trafficlight.Logic("L_34_EXTEND", 0, 0, [phases[2]])
##############################################################
##############################################################
##############################################################

phases = [] # ACTION 2
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_EW_S_30['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_EW_S_30['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_EW_S_30['green']))
LOGICS["EW_S_30_GREEN"] = traci.trafficlight.Logic("EW_S_30_GREEN",0, 0, [phases[0]])
LOGICS["EW_S_30_YELLOW"] = traci.trafficlight.Logic("EW_S_30_YELLOW", 0, 0,  [phases[1]])
# ACTION 6
LOGICS["EW_S_30_EXTEND"] = traci.trafficlight.Logic("EW_S_30_EXTEND", 0, 0,  [phases[2]])
##############################################################
phases = [] # ACTION 3
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_EW_L_30['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_EW_L_30['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_EW_L_30['green']))
LOGICS["EW_L_30_GREEN"] = traci.trafficlight.Logic("EW_L_30_GREEN",0, 0, [phases[0]])
LOGICS["EW_L_30_YELLOW"] = traci.trafficlight.Logic("EW_L_30_YELLOW", 0, 0,  [phases[1]])
# ACTION 7
LOGICS["EW_L_30_EXTEND"] = traci.trafficlight.Logic("EW_L_30_EXTEND", 0, 0,  [phases[2]])
##############################################################
phases = []  # ACTION 1
phases.append(traci.trafficlight.Phase(duration=GREEN_DURATION, state=phase_L_30['green']))
phases.append(traci.trafficlight.Phase(duration=YELLOW_DURATION, state=phase_L_30['yellow']))
phases.append(traci.trafficlight.Phase(duration=EXTEND_DURATION, state=phase_L_30['green']))
LOGICS["L_30_GREEN"] = traci.trafficlight.Logic("L_30_GREEN", 0, 0, [phases[0]])
LOGICS["L_30_YELLOW"] = traci.trafficlight.Logic("L_30_YELLOW", 0, 0, [phases[1]])
# ACTION 5
LOGICS["L_30_EXTEND"] = traci.trafficlight.Logic("L_30_EXTEND", 0, 0, [phases[2]])