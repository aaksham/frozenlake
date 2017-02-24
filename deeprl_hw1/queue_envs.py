# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.envs.registration import register
import numpy
import itertools

class QueueEnv(Env):
    """Implement the Queue environment from problem 3.

    Parameters
    ----------
    p1: float
      Value between [0, 1]. The probability of queue 1 receiving a new item.
    p2: float
      Value between [0, 1]. The probability of queue 2 receiving a new item.
    p3: float
      Value between [0, 1]. The probability of queue 3 receiving a new item.

    Attributes
    ----------
    nS: number of states
    nA: number of actions
    P: environment model
    """
    metadata = {'render.modes': ['human']}

    SWITCH_TO_1 = 0
    SWITCH_TO_2 = 1
    SWITCH_TO_3 = 2
    SERVICE_QUEUE = 3




    def __init__(self, p1, p2, p3):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(
            [(1, 3), (0, 5), (0, 5), (0, 5)])
        self.nS = 0
        self.nA = 4
        self.P = dict()
        self.current_state=(1,0,0,0)
        self.p1=p1
        self.p2=p2
        self.p3=p3


    def _reset(self):
        """Reset the environment.

        The server should always start on Queue 1.

        Returns
        -------
        (int, int, int, int)
          A tuple representing the current state with meanings
          (current queue, num items in 1, num items in 2, num items in
          3).
        """
        self.current_state=(1,0,0,0)
        return self.current_state

    def _step(self, action):
        """Execute the specified action.

        Parameters
        ----------
        action: int
          A number in range [0, 3]. Represents the action.

        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """
        possible_next_states=self.query_model(self.current_state,action)
        probarray=[]
        for ps in possible_next_states:
            probarray.append(ps[0])
        probs=numpy.asarray(probarray)
        randomarray=numpy.random.rand(len(possible_next_states),1)
        next_state_index=self.categorical_sample(probs,randomarray)
        pns=possible_next_states[next_state_index]
        next_state=(pns[1],pns[2],pns[3],dict())
        self.current_state=next_state[0]
        return next_state



    def _render(self, mode='human', close=False):
        print ("Current Q: "+str(self.current_state[0]))
        print ("Items in Q1: "+str(self.current_state[1]))
        print ("Items in Q2: "+str(self.current_state[2]))
        print ("Items in Q3: "+str(self.current_state[3]))
        print ("\n")


    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        pass

    def query_model(self, state, action):
        """Return the possible transition outcomes for a state-action pair.

        This should be in the same format at the provided environments
        in section 2.

        Parameters
        ----------
        state
          State used in query. Should be in the same format at
          the states returned by reset and step.
        action: int
          The action used in query.

        Returns
        -------
        [(prob, nextstate, reward, is_terminal), ...]
          List of possible outcomes
        """
        lst=list(itertools.product([0,1],repeat=3))
        reward=0
        newstate=list(state)
        if action==QueueEnv.SERVICE_QUEUE:
            currq=newstate[0]
            if newstate[currq]>0:
                newstate[currq]-=1
                reward=1
        elif action==QueueEnv.SWITCH_TO_1:
            newstate[0]=1
        elif action==QueueEnv.SWITCH_TO_2:
            newstate[0]=2
        elif action==QueueEnv.SWITCH_TO_3:
            newstate[0]=3
        blockq1=1
        blockq2=1
        blockq3=1
        if newstate[1]>=5: blockq1=0
        if newstate[2]>=5: blockq2=0
        if newstate[3]>=5: blockq3=0
        possible_states=[]
        for combination in lst:
            q1=combination[0]
            q2=combination[1]
            q3=combination[2]
            state_prob=0
            newpstate=newstate[:]
            if blockq1==0 or q1==0: state_prob+=(1-self.p1)
            else:
                state_prob+=self.p1
                newpstate[1]+=1
            if blockq2==0 or q2==0: state_prob=state_prob*(1-self.p2)
            else:
                state_prob=state_prob*self.p2
                newpstate[2]+=1
            if blockq3==0 or q3==0: state_prob=state_prob*(1-self.p3)
            else:
                state_prob=state_prob*self.p3
                newpstate[3]+=1
            found=False
            for psalready in possible_states:
                if tuple(newpstate) == psalready[1]:
                    found=True
                    break
            if not found: possible_states.append((state_prob,tuple(newpstate)))
        total_prob=0
        for ps in possible_states:
            total_prob+=ps[0]
        for i in range(len(possible_states)):
            unnormalized_state=possible_states[i]
            possible_states[i]=(float(unnormalized_state[0])/float(total_prob),unnormalized_state[1])
        final_list=[]
        for ps in possible_states:
            final_list.append((ps[0],ps[1],reward,False))
        return final_list

    def get_action_name(self, action):
        if action == QueueEnv.SERVICE_QUEUE:
            return 'SERVICE_QUEUE'
        elif action == QueueEnv.SWITCH_TO_1:
            return 'SWITCH_TO_1'
        elif action == QueueEnv.SWITCH_TO_2:
            return 'SWITCH_TO_2'
        elif action == QueueEnv.SWITCH_TO_3:
            return 'SWITCH_TO_3'
        return 'UNKNOWN'

    def categorical_sample(self, prob_n, np_random):
        """
        Sample from categorical distribution
        Each row specifies class probabilities
        """
        csprob_n = numpy.cumsum(prob_n)
        return (csprob_n > np_random).argmax()

register(
    id='Queue-1-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .9,
            'p3': .1})

register(
    id='Queue-2-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .1,
            'p3': .1})
