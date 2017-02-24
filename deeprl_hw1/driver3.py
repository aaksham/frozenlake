import deeprl_hw1.queue_envs as qenv
import numpy
P1 = 0.1
P2 = 0.9
P3 = 0.1

env=qenv.QueueEnv(P1,P2,P3)
#ps=env.query_model((1,0,0,0),1)
#print ps
ps=env.query_model((1,5,3,4),3)
print ps
numpy.random.seed(0)
env.reset()
env.render()
env._step(1)
env.render()
env._step(3)
env.render()
#
# ps=env.query_model((1,5,5,5),3)
# print ps
