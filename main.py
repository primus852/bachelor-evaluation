from pathlib import Path
import argparse
from absl import flags
import numpy as np
import pandas as pd
import time
import datetime
import statistics

import Agents
from Utils import Stack
from Utils import Helper

from pysc2 import maps
from pysc2.env import sc2_env
from pysc2.lib import features

if __name__ == '__main__':

    # Init the ArgParser
    parser = argparse.ArgumentParser(description='Start RL with SC2 Minigames')

    # Add Parser Arguments
    parser.add_argument("--version", help="Show Version", action="store_true")
    parser.add_argument("--visualize", help="Show Visualization Panel", action="store_true")
    parser.add_argument("--replay", help="Save Replay", action="store_false")
    parser.add_argument("--max_episodes", help="Max Episodes per Run", type=int, default=10000)
    parser.add_argument("--max_steps", help="Max Steps per Episode", type=int, default=400)
    parser.add_argument("--map", help="Map (without .SC2Map)", type=str, default='MoveToBeacon')
    parser.add_argument("--load_qtable", help="Load Qtable", action='store_true')
    parser.add_argument("--folder", help="Folder Path to save/load the models (relative to Agents Folder)", type=str,
                        default='models')

    args = parser.parse_args()

    # Check for Arguments
    if args.version:
        print("0.1.0")
        exit()

    FLAGS = flags.FLAGS
    FLAGS(['run_sc2'])
    steps = 0
    p = Path.cwd()

    x = 0
    qt = None
    states = None
    stack = Stack()
    stack_score = Stack()
    total_start_time = time.time()

    with sc2_env.SC2Env(
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            map_name=maps.get(args.map),
            realtime=False,
            agent_interface_format=[features.parse_agent_interface_format(
                feature_screen=16,
                feature_minimap=16,
                rgb_screen=None,
                rgb_minimap=None
            )],
            visualize=args.visualize) as env:

        if Path(p / args.folder / '{0}_qtable.npy'.format(args.map)).is_file() and Path(
                p / args.folder / '{0}_states.npy'.format(args.map)).is_file() and args.load_qtable:
            print('Loaded QTable and States from Files')
            qt = p / args.folder / '{0}_qtable.npy'.format(args.map)
            states = p / args.folder / '{0}_states.npy'.format(args.map)
        else:
            print('Started with fresh QTable')

        # Todo: Make Switch depending on Map
        agent = Agents.MoveToBeacon(qt, states)
        agent_name = Agents.MoveToBeacon.__name__

        stats_list = []
        for i in range(args.max_episodes):
            start_time = time.time()
            ep_reward = 0
            obs = env.reset()
            for j in range(args.max_steps):
                steps += 1
                state, action, func = agent.step(obs[0])
                obs = env.step(actions=[func])
                next_state, _ = agent.get_state(obs[0])
                reward = obs[0].reward
                ep_reward += reward
                loss = agent.qtable.update_qtable(state, next_state, action, reward)

            # Time it
            elapsed = time.time() - start_time
            episodes_left = args.max_episodes - i

            # IDEA: END IF THE STACK(100) LAST SCORE MEANS 26
            stack_score.push(ep_reward)
            if stack_score.size() > 100:
                stack_score.pop()

            mean_score = statistics.mean(stack_score.items)

            # Add time to Stack
            stack.push(elapsed)
            if stack.size() > 10:
                stack.pop()

            # Now calc the average of the stack size
            seconds_left = episodes_left * statistics.mean(stack.items)

            stats = {
                'Episode': i,
                'Reward': ep_reward,
                'Seconds': round(elapsed, 2),
                'Mean': round(mean_score, 2)
            }
            stats_list.append(stats)

            print('Ep.[{0}] Reward: {1} (Mean: {2}) in {3} seconds, {4} episodes left ({5}), Steps {6}, Threshold: {7}, Loss {8}'
                  .format(i, ep_reward, round(mean_score, 2), round(elapsed, 2), episodes_left,
                          str(datetime.timedelta(seconds=round(seconds_left))),agent.get_steps(), Helper.get_eps_threshold(0.9,0.025,2500,steps), loss))
            if args.replay and mean_score >= 20:
                env.save_replay(agent_name, '{0}_{1}'.format(agent_name, str(ep_reward)))

            # If we reach a mean Score of 26, exit the loop
            if mean_score >= 26:
                break

    # Save the Model / States
    df = pd.DataFrame(stats_list, columns=['Episode', 'Reward', 'Seconds', 'Mean'])
    try:
        df.to_csv(p / args.folder / '{0}_{1}.csv'.format(args.map, time.time()))
        agent.qtable.save_qtable(p / args.folder / '{0}_qtable.npy'.format(args.map))
        agent.qtable.save_states(p / args.folder / '{0}_states.npy'.format(args.map))
        np.load(p / args.folder / '{0}_states.npy'.format(args.map))
    except Exception as e:
        print('Error saving Files, Message: {0}'.format(e))
        exit()

    print('----------------------')
    print('MEAN SCORE {0} REACHED | EPISODES {1} | TIME NEEDED {2}'.format(mean_score, i, str(
        datetime.timedelta(seconds=round(time.time() - total_start_time)))))
    print('----------------------')
