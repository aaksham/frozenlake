# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import math

def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray
      The value for the given policy
    """
    value_func_old = np.random.rand(env.nS)
    value_func_new = np.zeros(env.nS)
    for iteration in range(max_iterations):
        delta=0
        for s in range(env.nS):
            vs=0
            actions=[policy[s]]
            #if len(actions)==1: actions=[actions]
            for a in actions:
                for possible_next_state in env.P[s][a]:
                    prob_action = possible_next_state[0]
                    cur_reward=possible_next_state[2]
                    future_reward=gamma*value_func_old[possible_next_state[1]]
                    vs+=prob_action*(cur_reward+future_reward)
                #if env.P[s][a][3]:break
            diff=abs(value_func_old[s]-vs)
            delta=max(delta,diff)
            value_func_new[s]=vs
        #delta=math.sqrt(delta)
        if delta<=tol: break
        value_func_old = value_func_new
    return value_func_new, iteration


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    policy=np.zeros(env.nS,dtype='int')
    for s in range(env.nS):
        maxvsa=-1
        maxa=-1
        for a in range(env.nA):
            vsa=0
            for possible_next_state in env.P[s][a]:
                prob_action = possible_next_state[0]
                cur_reward = possible_next_state[2]
                future_reward = gamma * value_function[possible_next_state[1]]
                vsa+=prob_action * (cur_reward + future_reward)
            if vsa>maxvsa:
                maxvsa=vsa
                maxa=a
        policy[s]=maxa

    return policy


def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    stable=True
    for s in range(env.nS):
        old_action=policy[s]
        maxvsa=-1
        maxa=-1
        for a in range(env.nA):
            vsa=0
            for possible_next_state in env.P[s][a]:
                prob_action = possible_next_state[0]
                cur_reward = possible_next_state[2]
                future_reward = gamma * value_func[possible_next_state[1]]
                vsa+=prob_action * (cur_reward + future_reward)
            if vsa>maxvsa:
                maxvsa=vsa
                maxa=a
        if maxa!=old_action: stable=False
        policy[s]=maxa
    return stable, policy


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    stable=False
    iters=0
    eval_iters=0
    while not stable:
        value_func,iter=evaluate_policy(env,gamma,policy)
        eval_iters+=iter
        stable,policy=improve_policy(env,gamma,value_func,policy)
        iters+=1
    return policy, value_func, iters, eval_iters


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func_old = np.random.rand(env.nS)
    value_func_new = np.zeros(env.nS)
    for iteration in range(max_iterations):
        delta=0
        for s in range(env.nS):
            maxvsa = -1
            for a in range(env.nA):
                vsa=0
                for possible_next_state in env.P[s][a]:
                    prob_action = possible_next_state[0]
                    cur_reward=possible_next_state[2]
                    if possible_next_state[3]:
                        future_reward=0
                    else: future_reward=gamma*value_func_old[possible_next_state[1]]
                    vsa+=prob_action*(cur_reward+future_reward)
                if vsa>maxvsa:
                    maxvsa=vsa
            #diff=math.pow((value_func_old[s]-maxvsa),2)
            diff=abs(value_func_old[s]-maxvsa)
            delta=max(delta,diff)
            value_func_new[s]=maxvsa
        #delta=math.sqrt(delta)
        if delta<=tol: break
        value_func_old = value_func_new

    return value_func_new, iteration


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)
    return str_policy
