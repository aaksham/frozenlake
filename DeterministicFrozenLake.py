import deeprl_hw1.lake_envs as lake_env
import gym
import time
import seaborn
from tabulate import tabulate
import matplotlib.pyplot as plt
from deeprl_hw1.rl1 import *

def run_policy(env,gamma,policy):
    initial_state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    current_state=initial_state
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(policy[current_state])
        env.render()

        total_reward += math.pow(gamma,num_steps)*reward
        num_steps += 1

        if is_terminal:
            break

        current_state=nextstate
        time.sleep(1)

    return total_reward, num_steps

grid=8
envname='Deterministic-'+str(grid)+'x'+str(grid)+'-FrozenLake-v0'
env = gym.make(envname)
env.render()
gamma=0.9
print "Executing Policy Iteration"
start_time=time.time()
policy, value_func, policy_iters, val_iters= policy_iteration(env,gamma)
print "Total time taken: "+str((time.time()-start_time))
print "Total Policy Improvement Steps: "+str(policy_iters)
print "Total Policy Evaluation Steps: "+str(val_iters)
print "Policy:"
policy_str=print_policy(policy,lake_env.action_names)
ps=[]
for elem in policy_str:
    ps.append(elem[0])
reshaped_policy=np.reshape(ps,(grid,grid))
print tabulate(reshaped_policy,tablefmt='latex')
f, ax = plt.subplots(figsize=(11, 9))
cmap = seaborn.diverging_palette(220, 10, as_cmap=True)
reshaped=np.reshape(value_func,(grid,grid))
seaborn.heatmap(reshaped, cmap=cmap, vmax=1.1,
            square=True, xticklabels=grid+1, yticklabels=grid+1,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.savefig('1c.png',bbox_inches='tight')
np.savetxt('1gpolicy.csv',reshaped,delimiter=',')

print "Executing Value Iteration"
start_time=time.time()
value_function,value_iters=value_iteration(env,gamma)
print "Total time taken: "+str((time.time()-start_time))
print "Total Value Iteration Steps: "+str(value_iters)
print "Policy:"
policy=value_function_to_policy(env,gamma,value_function)
policy_str=print_policy(policy,lake_env.action_names)
ps=[]
for elem in policy_str:
    ps.append(elem[0])
reshaped_policy=np.reshape(ps,(grid,grid))
print tabulate(reshaped_policy,tablefmt='latex')
f, ax = plt.subplots(figsize=(11, 9))
cmap = seaborn.diverging_palette(220, 10, as_cmap=True)
reshaped=np.reshape(value_function,(grid,grid))
seaborn.heatmap(reshaped, cmap=cmap, vmax=1.1,
            square=True, xticklabels=grid+1, yticklabels=grid+1,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.savefig('1e.png',bbox_inches='tight')
np.savetxt('1gvalue.csv',reshaped,delimiter=',')

cum_reward,nsteps=run_policy(env,gamma,policy)
print "Cumulative Reward: "+str(cum_reward)
print "No. of steps: "+str(nsteps)

