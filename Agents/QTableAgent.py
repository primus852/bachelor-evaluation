from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from QTable import QTable
import math
import random

import numpy as np

_AI_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_AI_SELECTED = features.SCREEN_FEATURES.selected.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_MOVE_RAND = 1000
_MOVE_MIDDLE = 2000
_BACKGROUND = 0
_AI_SELF = 1
_AI_ALLIES = 2
_AI_NEUTRAL = 3
_AI_HOSTILE = 4
_ARMY_SUPPLY = 5
_TERRAN_MARINE = 48
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

ACTION_DO_NOTHING = 'donothing'
ACTION_ATTACK_SINGLE = 'attack'
ACTION_ATTACK_ALL = 'attackall'

action_space = [
    ACTION_DO_NOTHING,
]

# Create actions for moving to any point on a 16 by 16 grid
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            action_space.append(ACTION_ATTACK_SINGLE + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))
            action_space.append(ACTION_ATTACK_ALL + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))


class QTableAgent(base_agent.BaseAgent):
    def __init__(self, load_qt=None, load_st=None):
        super(QTableAgent, self).__init__()
        self.qtable = QTable(action_space, load_qt=load_qt, load_st=load_st)
        self.steps = 0
        self.move_number = 0

    @staticmethod
    def split_action(action_id):
        smart_action = action_space[action_id]
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return smart_action, x, y

    def step(self, obs):
        """Step function gets called automatically by pysc2 environment"""
        super(QTableAgent, self).step(obs)
        state = self.get_state(obs)
        action = self.qtable.get_action(state, self.steps)
        smart_action, x, y = self.split_action(action)
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]

        # Default Return
        func = actions.FunctionCall(_NO_OP, [])

        if self.move_number == 0:
            self.move_number = 1

            # Randomly select a marine to attack with
            if smart_action == ACTION_ATTACK_SINGLE:
                unit_y, unit_x = (unit_type == _TERRAN_MARINE).nonzero()

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]

                    func = actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            # Select all marines to attack with
            elif smart_action == ACTION_ATTACK_ALL:
                if _SELECT_ARMY in obs.observation['available_actions']:
                    func = actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        elif self.move_number == 1:
            self.move_number = 0

            if smart_action == ACTION_ATTACK_SINGLE or smart_action == ACTION_ATTACK_ALL:

                if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)

                    func = actions.FunctionCall(_ATTACK_MINIMAP,
                                                [_NOT_QUEUED, (int(x) + (x_offset * 4), int(y) + (y_offset * 4))])

        return state, action, func

    @staticmethod
    def get_state(obs):

        # Quantize the current state to sixteen squares to reduce action space, also keep track of
        # the number of marines
        current_state = np.zeros(17)
        current_state[0] = obs.observation['player'][_ARMY_SUPPLY]

        # Make array of "hot squares" indicating locations of enemy (or neutral) units/beacons/shards
        hot_squares = np.zeros(16)
        enemy_y, enemy_x = (obs.observation['feature_minimap'][_AI_RELATIVE] == _AI_HOSTILE).nonzero()
        neutral_y, neutral_x = (obs.observation['feature_minimap'][_AI_RELATIVE] == _AI_NEUTRAL).nonzero()

        enemy_y = np.concatenate((enemy_y, neutral_y))
        enemy_x = np.concatenate((enemy_x, neutral_x))

        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))

            hot_squares[((y - 1) * 4) + (x - 1)] = 1

        for i in range(0, 16):
            current_state[i + 1] = hot_squares[i]

        return tuple(current_state)

    def get_steps(self):
        return self.steps
