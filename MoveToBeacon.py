import logging
import os
import random
import json
import time
from datetime import datetime
import uuid

import pandas as pd
import pathlib

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from QLearning import QLearning

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

"""Define all the choices the Agent can make"""
AVAILABLE_CHOICES = [
    'do_nothing',
    'move_marine'
]


class MoveToBeacon(base_agent.BaseAgent):
    def __init__(self, model, counter):

        """Init all needed Vars"""
        super(MoveToBeacon, self).__init__()

        """Init the Timer"""
        self.start_time = datetime.now()

        """ Init the model"""
        self.model = model
        self.counter = counter

        """Init the Logger"""
        self.logger = logging.getLogger(__name__)
        self.logger.info('Starting the Agent, using %s Model' % self.model)

        """
        Init the Result List
        ToDo: Implement Count Logic for NEXUS Counter, as "expand_now" may fail
        """
        self.result_list = {
            'ID': str(uuid.uuid4()),
            'GAME_MINUTES': 0,
            'REWARD': 0,
            'PROCESS_SECONDS': 0,
            'AVG_QLEARN_CHOOSE_SECONDS': 0,
            'AVG_QLEARN_LEARN_SECONDS': 0,
            'ADDED_STATES': 0,
            'CUMUL_STATES': 0,
        }

        if self.model != 'qlearn' and self.model != 'nn':
            self.logger.exception('Invalid Model selected')
            exit()

        self.save_path = pathlib.Path(__file__).parent.parent.parent / QLEARN_FILE
        json_file = '%sresults_%s.json' % (JSON_RESULTS, time.time())
        self.result_file = pathlib.Path(__file__).parent.parent.parent / json_file

        self.qlearn = None
        self.qlearn_size = None
        if self.model == 'qlearn':
            self.qlearn = QLearning(actions=list(range(len(AVAILABLE_CHOICES))))
            """Read Pickle"""
            if os.path.isfile(self.save_path):
                os.chmod(self.save_path, 0o755)
                self.qlearn.q_table = pd.read_pickle(self.save_path, compression='gzip')

            """Count the Length"""
            self.qlearn_size = len(self.qlearn.q_table)

        """Pause until this counter has passed"""
        self.continue_after = 0

        """Minutes of game time"""
        self.minutes = 0
        self.total_steps = 0

        """Scout IDs"""
        self.scouts_spots = {}

        """Init the rewards"""
        self.killed_units = 0
        self.killed_buildings = 0

        """Init first previous action"""
        self.previous_action = None
        self.previous_state = None

    def step(self, obs):

        """Step function gets called automatically by pysc2 environment"""
        super(MoveToBeacon, self).step(obs)

        """Get total Steps"""
        self.total_steps += 1

        """Get the required states"""
        current_state = self.get_state()

        if self.previous_action is not None:
            reward = 0

            self.result_list['REWARD'] += reward
            if self.model == 'qlearn':
                start_time_learn = datetime.now()
                self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))
                stop_time_learn = datetime.now()
                time_learn = stop_time_learn - start_time_learn
                # print('Learn Time: %s, Length: %s' % (round(time_learn.total_seconds(), 4), len(self.qlearn.q_table)))
                self.result_list['AVG_QLEARN_LEARN_SECONDS'] += time_learn.total_seconds()

        """Choose next action based on model"""
        if self.minutes > self.continue_after:
            if self.model == 'qlearn':
                start_time_choose = datetime.now()
                next_action = self.qlearn.choose_action(str(current_state))
                stop_time_choose = datetime.now()
                time_choose = stop_time_choose - start_time_choose
                self.result_list['AVG_QLEARN_CHOOSE_SECONDS'] += time_choose.total_seconds()
            else:
                next_action = random.choice(AVAILABLE_CHOICES)

            self.previous_state = current_state
            self.previous_action = next_action

            try:
                # self.logger.info('%s' % AVAILABLE_CHOICES[next_action])
                await getattr(self, AVAILABLE_CHOICES[next_action])()
            except Exception as e:
                print(next_action)
                print(self.qlearn.q_table)
                self.logger.exception('Invalid Choice: ' + AVAILABLE_CHOICES[next_action])
                exit()

    def on_end(self, game_result: Result):
        """Get the final State"""
        current_state = self.get_state()

        if game_result == Result.Victory:
            reward = REWARD_WON_GAME
            self.result_list['VICTORY'] = 1
        elif game_result == Result.Defeat:
            reward = REWARD_LOST_GAME
            self.result_list['DEFEAT'] = 1
        else:
            reward = REWARD_TIE_GAME
            self.result_list['TIE'] = 1

        self.result_list['REWARD'] += reward
        self.result_list['GAME_MINUTES'] = self.time_formatted

        """Learn one last time"""
        if self.model == 'qlearn':
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

            """Save to File"""
            self.qlearn.q_table.to_pickle(self.save_path, 'gzip')

        """Stop the Timer"""
        stop_time = datetime.now()
        elapsed = stop_time - self.start_time
        self.result_list['PROCESS_SECONDS'] = elapsed.total_seconds()

        """Avg. of QLEARN Seconds"""
        self.result_list['AVG_QLEARN_CHOOSE_SECONDS'] = self.result_list['AVG_QLEARN_CHOOSE_SECONDS'] / self.total_steps
        self.result_list['AVG_QLEARN_LEARN_SECONDS'] = self.result_list['AVG_QLEARN_LEARN_SECONDS'] / self.total_steps

        """Added Qlearn Rows"""
        self.result_list['ADDED_STATES'] = len(self.qlearn.q_table) - self.qlearn_size
        self.result_list['CUMUL_STATES'] = len(self.qlearn.q_table)

        """Save to JSON File"""
        with open(self.result_file, 'w') as outfile:
            json.dump(self.result_list, outfile)

        """ Exit the Logger"""
        logging.shutdown()
        del self.logger

    @staticmethod
    def scale_down(value, resource):
        if resource == 'mineral':
            max_val = 1500
            medium_val = 500
        else:
            max_val = 500
            medium_val = 150

        if value >= max_val:
            return 100
        elif medium_val <= value < max_val:
            return 50
        elif 0 < value < medium_val:
            return 10
        else:
            return 0

    def get_state(self):

        # Scale Vespene and Minerals in low, medium, high
        minerals = self.scale_down(self.minerals, 'minerals')
        vespene = self.scale_down(self.vespene, 'gas')

        """Gather the current state"""
        current_state = [
            len(self.workers),  # All Probes
            len(self.geysers),  # All Assimilators
            len(self.units(GATEWAY).ready),  # All Gateways
            len(self.units(STARGATE).ready),  # All Stargates
            len(self.units(ROBOTICSFACILITY).ready),  # All Robotics
            len(self.units(DARKSHRINE).ready),  # All Dark Shrines
            len(self.townhalls),  # All Nexuses
            self.supply_used,  # Used Supply
            self.supply_cap,  # Max Supply
            self.supply_left,  # Supply Left
            minerals,  # Scaled Minerals
            vespene,  # Scaled Vespene
            len(self.units(ZEALOT).ready),  # All Zealots
            len(self.units(STALKER).ready),  # All Stalker
            len(self.units(VOIDRAY).ready),  # All Voidrays
            len(self.units(IMMORTAL).ready),  # All Immortals
            len(self.units(DARKTEMPLAR).ready),  # All Dark Templars
        ]

        return current_state

    async def build_building(self, building):
        """
        Build any building near a random pylon, facing the center of the map
        If there are no pylons, build one
        :param building:
        :return:
        """
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.noqueue.random
            if self.can_afford(building) and not self.already_pending(building):
                self.result_list[building.name] += 1
                await self.build(building, near=pylon.position.towards(self.game_info.map_center, 5))
        else:
            await self.build_pylon()

    async def build_unit(self, unit, buildings):
        """
        Build a unit in a random building with space in queue that can create this unit
        :param unit:
        :param buildings:
        :return:
        """
        if buildings.noqueue.exists:
            if self.can_afford(unit):
                self.result_list[unit.name] += 1
                await self.do(random.choice(buildings.noqueue).train(unit))

    async def do_nothing(self):
        """Do nothing for 0.07 to 1 second"""
        wait = random.randrange(7, 100) / 100
        self.result_list['TOTAL_WAIT'] += wait
        self.continue_after = self.minutes + wait

    async def build_probe(self):
        """
        Choice 1: Build a probe
        Check if there is 1+ Nexus (with queue available), if so and enough Resources available,
        build worker on a random nexus
        """
        nexuses = self.units(NEXUS).ready.noqueue
        if nexuses.exists:
            await self.build_unit(PROBE, nexuses)

    async def build_pylon(self):
        """
        Choice 2: Build a pylon
        Check if there is 1+ Nexus, if so and enough Resources available, build a pylon near random nexus, towards the
        center of the map
        ToDo: Check if we can make this completely random, may enable cheese tactics?
        """
        nexuses = self.units(NEXUS).ready
        if nexuses.exists:
            if self.can_afford(PYLON) and not self.already_pending(PYLON):
                self.result_list[PYLON.name] += 1
                await self.build(PYLON, near=self.units(NEXUS).first.position.towards(self.game_info.map_center, 5))

    async def build_assimilator(self):
        """
        Choice 3: Build an assimilator to harvest gas
        Check for gas within 15 units of every Nexus
        For every geysir found check if we have a probe and if none i build already, build the assimilator
        """
        for nexus in self.units(NEXUS).ready:
            gases = self.state.vespene_geyser.closer_than(15.0, nexus)
            for gas in gases:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(gas.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, gas).exists:
                    self.result_list[ASSIMILATOR.name] += 1
                    await self.do(worker.build(ASSIMILATOR, gas))

    async def build_gateway(self):
        """
        Build a gateway
        """
        await self.build_building(GATEWAY)

    async def build_cybernetics(self):
        """
        Build a cybernetics core (only one)
        """
        gateways = self.units(GATEWAY).ready
        cybernetics_cores = self.units(CYBERNETICSCORE)

        if gateways.exists and not cybernetics_cores.exists:
            await self.build_building(CYBERNETICSCORE)

    async def build_stargate(self):
        """
        Build a stargate if there is a cybernetics core
        """
        if self.units(CYBERNETICSCORE).ready.exists:
            await self.build_building(STARGATE)

    async def build_dark_shrine(self):
        """
        Build a dark shrine (only one) if there is a cybernetics core and a twilight council
        """
        if self.units(CYBERNETICSCORE).ready.exists and self.units(TWILIGHTCOUNCIL).ready.exists and not self.units(
                DARKSHRINE).exists:
            await self.build_building(DARKSHRINE)

    async def build_twilight_council(self):
        """
        Build a Twilight Council (only one) if there is a cybernetics core
        """
        if self.units(CYBERNETICSCORE).ready.exists and not self.units(TWILIGHTCOUNCIL).exists:
            await self.build_building(TWILIGHTCOUNCIL)

    async def build_robotics(self):
        """
        Build a robotics facility if a cybernetics core is available
        """
        if self.units(CYBERNETICSCORE).ready.exists:
            await self.build_building(ROBOTICSFACILITY)

    async def build_forge(self):
        """
        Build a Forge (only one)
        """
        if not self.units(FORGE).exists:
            await self.build_building(FORGE)

    async def build_cannon(self):
        """
        Build a Photon Cannon if a forge exists
        """
        if self.units(FORGE).ready.exists:
            await self.build_building(PHOTONCANNON)

    async def build_zealot(self):
        """
        Build a Zealot if there are ready Gateways
        """
        gateways = self.units(GATEWAY).ready

        if gateways.exists:
            await self.build_unit(ZEALOT, gateways)

    async def build_stalker(self):
        """
        Build a Stalker if there are ready Gateways and a Cybernetics Core
        """
        gateways = self.units(GATEWAY).ready
        cybernetics_cores = self.units(CYBERNETICSCORE).ready

        if gateways.exists and cybernetics_cores.exists:
            await self.build_unit(STALKER, gateways)

    async def build_voidray(self):
        """
        Build a voidray if there are Stargates and a Cybernetics Core
        """
        stargates = self.units(STARGATE).ready
        cybernetics_cores = self.units(CYBERNETICSCORE).ready

        if stargates.exists and cybernetics_cores.exists:
            await self.build_unit(VOIDRAY, stargates)

    async def build_immortal(self):
        """
        Build an immortal there are Robotics Facilities and a Cybernetics Core
        """
        robotics = self.units(ROBOTICSFACILITY).ready
        cybernetics_cores = self.units(CYBERNETICSCORE).ready

        if robotics.exists and cybernetics_cores.exists:
            await self.build_unit(IMMORTAL, robotics)

    async def build_templar(self):
        """
        Build a dark templar if there is a Dark Shrine and Gateways
        Todo: Check if DT can still be build if CYBERNETICSCORE and/or TWILIGHTCOUNCIL are destroyed (Source: Stefan)
        """
        dark_shrines = self.units(DARKSHRINE).ready
        gateways = self.units(GATEWAY).ready

        if dark_shrines.exists and gateways.exists:
            await self.build_unit(DARKTEMPLAR, gateways)

    async def action_attack_enemy_unit(self):
        """
        Attack known enemy unit
        If there a visible enemy units and we have idle units, use them to attack
        :return:
        """
        if len(self.known_enemy_units) > 0:
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                target = self.known_enemy_units.closest_to(random.choice(nexuses))
                for unit in AVAILABLE_ARMY:
                    for u in self.units(unit).idle:
                        await self.do(u.attack(target))

    async def action_attack_enemy_building(self):
        """
        Attack known enemy building
        If there a known enemy structures and we have idle units, use them to attack
        :return:
        """
        if len(self.known_enemy_structures) > 0:
            target = random.choice(self.known_enemy_structures)
            for unit in AVAILABLE_ARMY:
                for u in self.units(unit).idle:
                    await self.do(u.attack(target))

    async def action_defend(self):
        """
        Defend own Nexus
        :return:
        """
        if len(self.known_enemy_units) > 0:
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                target = self.known_enemy_units.closest_to(random.choice(nexuses))
                for unit in AVAILABLE_ARMY:
                    for u in self.units(unit).idle:
                        await self.do(u.attack(target))

    async def action_expand(self):
        """
        Expand to another Location
        :return:
        """
        if self.can_afford(NEXUS):
            await self.expand_now()

    async def action_scout(self):
        """
        Send a scouting probe
        Variation: If there is a Robotics Facility, see if we can use/build an observer
        :return:
        """

        """List of Expansions"""
        exp_list = {}

        """
        Measure distance to all of the Expansions
        Note: this needs to be changed if it is used on a map for >2 Players
        """
        for el in self.expansion_locations:
            distance_to_enemy = el.distance_to(self.enemy_start_locations[0])
            exp_list[distance_to_enemy] = el

        """Order by Distance"""
        ordered_exp_distances = sorted(k for k in exp_list)

        """Ids of Units"""
        existing_ids = [unit.tag for unit in self.units]

        """List of Units that are dead already"""
        dead_units = []
        for noted_scout in self.scouts_spots:
            if noted_scout not in existing_ids:
                dead_units.append(noted_scout)

        """Remove from Scouts List id in dead list"""
        for scout in dead_units:
            del self.scouts_spots[scout]

        """See if we have a ROBOTICSFACILITY, if not use a PROBE"""
        robotics = self.units(ROBOTICSFACILITY).ready
        observers = self.units(OBSERVER).ready
        scout_unit = OBSERVER if robotics.exists else PROBE

        """If there is no OBSERVER, but a ROBOTICSFACILITY, build an OBSERVER and stop the scouting"""
        if robotics.exists and not observers.exists:
            self.result_list[OBSERVER.name] += 1
            await self.build_unit(OBSERVER, robotics)

        """Only make one PROBE a scout"""
        assign_scout = True
        if scout_unit == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scouts_spots:
                    assign_scout = False

        """Send to Expansion until it dies or has everything discovered"""
        if assign_scout and len(self.units(scout_unit).idle) > 0:
            for obs in self.units(scout_unit).idle[:1]:
                if obs.tag not in self.scouts_spots:
                    for dist in ordered_exp_distances:
                        try:
                            location = next(value for key, value in exp_list.items() if key == dist)
                            active_locations = [self.scouts_spots[k] for k in self.scouts_spots]

                            if location not in active_locations:
                                if scout_unit == PROBE:
                                    for unit in self.units(PROBE):
                                        if unit.tag in self.scouts_spots:
                                            continue

                                self.result_list['SENT_SCOUT'] += 1
                                await self.do(obs.move(location))
                                self.scouts_spots[obs.tag] = location
                                break
                        except Exception as e:
                            pass
