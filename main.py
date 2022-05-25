"""
Main function
Type in terminal to see writer in web page: tensorboard --logdir=SavedLoss

CUDA version: 11.2
tensorboard: 2.9.0
"""
import time
import torch as T
from functions import *
from check_state import CheckState
from buffer import MultiAgentReplayBuffer
from matd3 import MATD3
from make_env import MultiAgentEnv
from normalization import NormalizeData
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    # Check threads (CPU):
    # print("the number of cpu threads: {}".format(T.get_num_threads()))
    # Check CUDA
    # T.cuda.is_available()
    # Check device: GPUx or CPU
    # T.cuda.get_device_name(0)
    # Calculate program time
    # start = time.time()
    # print("time of each step: {}".format(time.time() - start))

    # True: train network; False: test network
    train_model = True
    # scenario = '1Ship'
    scenario = '2Ships_Cross'
    # scenario = '2Ships_Headon'
    # scenario = '3Ships_Cross&Headon'
    env = MultiAgentEnv(scenario)
    n_agents = env.ships_num
    actor_dims = env.ships_obs_space
    critic_dims = sum(actor_dims)

    # action space
    n_actions = env.ship_action_space

    chkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedNetwork'
    learn_interval = 100
    marl_agents = MATD3(chkpt_dir, actor_dims, critic_dims, n_agents, n_actions, learn_interval,
                        fc1=32, fc2=16, alpha=0.01, beta=0.01)
    max_size = 1000000
    memory = MultiAgentReplayBuffer(max_size, actor_dims, critic_dims, n_agents, n_actions, batch_size=1024)

    dis_redundant = 100
    dis_safe = 100
    dis_CPA1 = dis_safe * 5
    dis_CPA2 = dis_safe
    check_env = CheckState(env.ships_num, env.ships_pos, env.ships_term, env.ships_speed, env.ships_head,
                           env.angle_limit, dis_redundant, dis_safe, dis_CPA1, dis_CPA2)
    reward_max = check_env.reward_max

    norm_data = NormalizeData(env.ships_dis_max)

    steps_games = 10000  # number of maximum episodes
    steps_exp = steps_games / 2
    # a reasonable simulation time
    steps_max = (((env.ships_dis_max / env.ships_vel_min) // 500) + 1) * 500
    print('... the maximum simulation step in each episode:', steps_max[0], '...')
    steps_total = 0
    print_interval = 500

    score_history = []
    score_avg_history = []
    score_best = 0  # for saving path
    score_best_avg = 0  # for saving check points

    path_global = []
    result_dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedResult'
    rewards_global = []

    writer = SummaryWriter("SavedLoss")
    writer_dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedLoss'
    delete_files(writer_dir)
    if train_model:
        for i in range(steps_games + 1):
            obs = env.reset()
            # limits on ships' heading angles
            obs[:, 2] = warp_to_360(obs[:, 2], env.ships_num)
            n_obs = obs.copy()
            # data normalization
            for ship_idx in range(env.ships_num):
                n_obs[ship_idx, 0] = norm_data.nmlz_pos(obs[ship_idx, 0],
                                                        env.ships_x_min[ship_idx], env.ships_x_max[ship_idx])
                n_obs[ship_idx, 1] = norm_data.nmlz_pos(obs[ship_idx, 1],
                                                        env.ships_y_min[ship_idx], env.ships_y_max[ship_idx])
                n_obs[ship_idx, 2] = norm_data.nmlz_ang(obs[ship_idx, 2])
            done_reset = False
            done_goal = env.ships_done_term

            score = 0
            step_episode = 0
            if i < steps_exp:
                Exploration = True
                marl_agents.reset_noise()
            else:
                Exploration = False

            path_local = []
            rewards_local = []

            dis_closest = 0
            t_closest = 0
            dis_info = []
            # dis_info.append(env.ships_dis_max)
            while not done_reset:
                actions = marl_agents.choose_action(n_obs, Exploration)
                # list type, example: [-1.0, 1.0]
                obs_ = env.step(actions).copy()
                # For local observation problems, observations are not equal to states.
                # For global observation problems, observations are equal to states.
                # For simplification, just set: observations = states.
                state = obs.reshape(1, -1).copy()
                state_ = obs_.reshape(1, -1).copy()
                # reward
                reward_done, done_goal = check_env.check_done(obs_, done_goal)
                reward_term = check_env.check_term(obs_)
                if env.ships_num == 1:
                    reward = reward_term + reward_done
                else:
                    reward_coll, done_coll, dis_closest_local = check_env.check_coll(obs_)
                    reward_CORLEG, table = check_env.check_CORLEGs(obs, obs_)
                    reward_CPA = check_env.check_CPA(obs_)
                    reward = reward_term + reward_done + reward_coll + reward_CORLEG + reward_CPA
                # print(step_episode, table)
                # print(reward, reward_term, reward_coll, reward_CORLEG)
                rewards_local.append(reward)
                if env.ships_num == 1:
                    pass
                else:
                    if step_episode == 0:
                        dis_closest = dis_closest_local[0]
                    elif dis_closest < dis_closest_local[0]:
                        dis_closest = dis_closest_local[0]
                        t_closest = step_episode

                if step_episode >= steps_max:
                    done_reset = True
                if any(done_goal):
                    done_reset = True
                if env.ships_num > 1:
                    if done_coll:
                        done_reset = True

                # data normalization
                n_reward = reward.copy()
                n_obs_ = obs_.copy()
                for ship_idx in range(env.ships_num):
                    n_obs[ship_idx, 0] = norm_data.nmlz_pos(obs[ship_idx, 0],
                                                            env.ships_x_min[ship_idx], env.ships_x_max[ship_idx])
                    n_obs[ship_idx, 1] = norm_data.nmlz_pos(obs[ship_idx, 1],
                                                            env.ships_y_min[ship_idx], env.ships_y_max[ship_idx])
                    n_obs[ship_idx, 2] = norm_data.nmlz_ang(obs[ship_idx, 2])
                    n_obs_[ship_idx, 0] = norm_data.nmlz_pos(obs_[ship_idx, 0],
                                                             env.ships_x_min[ship_idx], env.ships_x_max[ship_idx])
                    n_obs_[ship_idx, 1] = norm_data.nmlz_pos(obs_[ship_idx, 1],
                                                             env.ships_y_min[ship_idx], env.ships_y_max[ship_idx])
                    n_obs_[ship_idx, 2] = norm_data.nmlz_ang(obs_[ship_idx, 2])

                    n_reward[ship_idx] = norm_data.nmlz_r(reward[ship_idx], reward_max)

                n_state = n_obs.reshape(1, -1)
                n_state_ = n_obs_.reshape(1, -1)
                memory.store_transition(n_obs, n_state, actions, n_reward, n_obs_, n_state_, done_goal)

                if not memory.ready():
                    pass
                else:
                    # See losses in writer, which can be shown in web page
                    # Type in terminal to see writer: tensorboard --logdir=SavedLoss
                    marl_agents.learn(memory, writer, steps_total)

                obs = obs_.copy()
                path_local.append(state)
                if env.ships_num == 1:
                    pass
                else:
                    dis_info.append([dis_closest_local])

                score += sum(reward)
                steps_total += 1
                step_episode += 1
            if i == 0:
                score_best = score
                # np.save(result_dir + '/path_first.npy', path_local)
                # np.save(result_dir + '/rewards_first.npy', rewards_global)
                # if env.ships_num == 1:
                #     pass
                # else:
                #     np.save(result_dir + '/info_closest_most_first.npy', [dis_closest, t_closest])
                #     np.save(result_dir + '/info_closest_all_first.npy', dis_info)
                print('... updating the optimal path group ( episode:', i, ', reward:', score_best, ') ...')
            elif score > score_best:
                score_best = score
                path_global = path_local
                rewards_global = rewards_local
                np.save(result_dir + '/path_global.npy', path_global)
                if env.ships_num == 1:
                    pass
                else:
                    np.save(result_dir + '/info_closest_most_global.npy', [dis_closest, t_closest])
                    np.save(result_dir + '/info_closest_all_global.npy', dis_info)
                print('... updating the optimal path group ( episode:', i, ', reward:', score_best, ') ...')


            score_history.append(score)
            score_avg = np.mean(score_history[-100:])
            score_avg_history.append(score)
            if score_avg > score_best_avg:
                marl_agents.save_checkpoint()
                score_best_avg = score_avg
            if i % print_interval == 0 and i > 0:
                print('episode:', i, ', the average score of last 100 episodes: {:.1f}'.format(score_avg))

        writer.close()
        # save networks
        marl_agents.save_checkpoint()
        # save data
        np.save(result_dir + '/score_history.npy', score_history)
        np.save(result_dir + '/score_avg_history.npy', score_avg_history)

        np.save(result_dir + '/path_last.npy', path_local)
        np.save(result_dir + '/rewards_global_last.npy', rewards_global)
        if env.ships_num == 1:
            pass
        else:
            np.save(result_dir + '/info_closest_most_last.npy', [dis_closest, t_closest])
            np.save(result_dir + '/info_closest_all_last.npy', dis_info)

    else:
        marl_agents.load_checkpoint()

        obs = env.reset()
        # limits on ships' heading angles
        obs[:, 2] = warp_to_360(obs[:, 2], env.ships_num)
        n_obs = obs.copy()
        # data normalization
        for ship_idx in range(env.ships_num):
            n_obs[ship_idx, 0] = norm_data.nmlz_pos(obs[ship_idx, 0],
                                                    env.ships_x_min[ship_idx], env.ships_x_max[ship_idx])
            n_obs[ship_idx, 1] = norm_data.nmlz_pos(obs[ship_idx, 1],
                                                    env.ships_y_min[ship_idx], env.ships_y_max[ship_idx])
            n_obs[ship_idx, 2] = norm_data.nmlz_ang(obs[ship_idx, 2])
        done_reset = False
        done_goal = env.ships_done_term

        Exploration = False

        path_test = []
        path_test.append(obs.reshape(1, -1))
        rewards_test = []

        step_episode = 0
        dis_closest = 0
        t_closest = 0
        dis_info = []
        while not done_reset:
            actions = marl_agents.choose_action(n_obs, Exploration)
            # print(actions)
            # list type, example: [-1.0, 1.0]
            obs_ = env.step(actions).copy()
            # print(obs_)

            # For local observation problems, observations are not equal to states.
            # For global observation problems, observations are equal to states.
            # For simplification, just set: observations = states.
            state = obs.reshape(1, -1)
            state_ = obs_.reshape(1, -1)

            # reward
            reward_done, done_goal = check_env.check_done(obs_, done_goal)
            reward_term = check_env.check_term(obs_)
            if env.ships_num == 1:
                reward = reward_term + reward_done
            else:
                reward_coll, done_coll, dis_closest_local = check_env.check_coll(obs_)
                reward_CORLEG, table = check_env.check_CORLEGs(obs, obs_)
                reward_CPA = check_env.check_CPA(obs_)
                reward = reward_term + reward_done + reward_coll + reward_CORLEG + reward_CPA
            # print(step_episode, table)
            # print(reward, reward_term, reward_coll, reward_CORLEG)
            rewards_test.append(reward)

            if env.ships_num == 1:
                pass
            else:
                if step_episode == 0:
                    dis_closest = dis_closest_local[0]
                elif dis_closest < dis_closest_local[0]:
                    dis_closest = dis_closest_local[0]
                    t_closest = step_episode

            if step_episode >= steps_max:
                done_reset = True
            # if all(done_goal):
            # if i == 0:
            #     print(i, state, done_goal)
            if any(done_goal):
                done_reset = True
            # if done_coll:
            #     done_reset = True

            # data normalization
            n_reward = reward.copy()
            n_obs_ = obs_.copy()
            for ship_idx in range(env.ships_num):
                n_obs_[ship_idx, 0] = norm_data.nmlz_pos(obs_[ship_idx, 0],
                                                         env.ships_x_min[ship_idx], env.ships_x_max[ship_idx])
                n_obs_[ship_idx, 1] = norm_data.nmlz_pos(obs_[ship_idx, 1],
                                                         env.ships_y_min[ship_idx], env.ships_y_max[ship_idx])
                n_obs_[ship_idx, 2] = norm_data.nmlz_ang(obs_[ship_idx, 2])

                n_reward[ship_idx] = norm_data.nmlz_r(reward[ship_idx], reward_max)
            n_state = n_obs.reshape(1, -1)
            n_state_ = n_obs_.reshape(1, -1)

            obs = obs_.copy()
            path_test.append(state)
            rewards_test.append(reward)
            step_episode += 1

        # save data
        np.save(result_dir + '/path_test.npy', path_test)
        np.save(result_dir + '/rewards_test.npy', rewards_test)
        if env.ships_num == 1:
            pass
        else:
            np.save(result_dir + '/info_closest_global_test.npy', [dis_closest, t_closest])
            np.save(result_dir + '/info_closest_local_test.npy', dis_info)
