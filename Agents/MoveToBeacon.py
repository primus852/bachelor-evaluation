from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from QTable import QTable
from QLearning import QLearning
from pprint import pprint

import numpy as np

_AI_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_AI_SELECTED = features.SCREEN_FEATURES.selected.index
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_MOVE_RAND = 1000
_MOVE_MIDDLE = 2000
_BACKGROUND = 0
_AI_SELF = 1
_AI_ALLIES = 2
_AI_NEUTRAL = 3
_AI_HOSTILE = 4
_SELECT_ALL = [0]
_NOT_QUEUED = [0]
_QUEUED = [1]

action_space = [
    _NO_OP,
    _SELECT_ARMY,
    _SELECT_POINT,
    # _MOVE_SCREEN,
    _MOVE_RAND,
    _MOVE_MIDDLE
]


class MoveToBeacon(base_agent.BaseAgent):
    def __init__(self, load_qt=None, load_st=None):
        super(MoveToBeacon, self).__init__()
        self.qtable = QTable(action_space, load_qt=load_qt, load_st=load_st)
        self.steps = 0

    def step(self, obs):
        """Step function gets called automatically by pysc2 environment"""
        super(MoveToBeacon, self).step(obs)
        state, beacon_pos = self.get_state(obs)
        action = self.qtable.get_action(state, self.steps)
        func = actions.FunctionCall(_NO_OP, [])

        if action_space[action] == _NO_OP:
            """Do nothing"""
            # print('Do Nothing')
            func = actions.FunctionCall(_NO_OP, [])

        elif action_space[action] == _SELECT_ARMY:
            """Select the Marine"""
            # print('Select Marine')
            func = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])

        elif state[0] and action_space[action] == _SELECT_POINT:
            """Move to a point"""
            ai_view = obs.observation['feature_screen'][_AI_RELATIVE]
            backgroundxs, backgroundys = (ai_view == _BACKGROUND).nonzero()
            point = np.random.randint(0, len(backgroundxs))
            backgroundx, backgroundy = backgroundxs[point], backgroundys[point]
            # print('Move Choice {0},{1}'.format(backgroundy, backgroundx))
            func = actions.FunctionCall(_SELECT_POINT, [_QUEUED, [backgroundy, backgroundx]])

        elif state[0] and action_space[action] == _MOVE_RAND:
            "Move somewhere random"
            movex, movey = np.random.randint(0, 32), np.random.randint(0, 32)
            # print('Move Random {0},{1}'.format(movex, movey))
            func = actions.FunctionCall(_MOVE_SCREEN, [_QUEUED, [movey, movex]])

        return state, action, func

    @staticmethod
    def get_state(obs):
        # get the positions of the marine and the beacon
        ai_view = obs.observation['feature_screen'][_AI_RELATIVE]
        beaconxs, beaconys = (ai_view == _AI_NEUTRAL).nonzero()
        marinexs, marineys = (ai_view == _AI_SELF).nonzero()
        marinex, mariney = marinexs.mean(), marineys.mean()

        marine_on_beacon = np.min(beaconxs) <= marinex <= np.max(beaconxs) and np.min(beaconys) <= mariney <= np.max(
            beaconys)

        # get a 1 or 0 for whether or not our marine is selected
        ai_selected = obs.observation['feature_screen'][_AI_SELECTED]
        marine_selected = int((ai_selected == 1).any())

        # print('Selected: {0} | On Beacon: {1} | Beacon Pos: {2},{3}'.format(marine_selected, int(marine_on_beacon), beaconxs, beaconys))
        # print('State {0}, {1}'.format((marine_selected, int(marine_on_beacon)), [beaconxs, beaconys]))

        return (marine_selected, int(marine_on_beacon)), [beaconxs, beaconys]
