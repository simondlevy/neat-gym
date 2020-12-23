'''
XOR evaluator for NEAT

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import numpy as np
import neat

def eval_xor(genome, config):
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    sse = 0

    for inp,tgt in zip(((0,0), (0,1), (1,0), (1,1)), (0,1,1,0)):
        sse += (tgt - net.activate(inp + (1,))[0])**2

    return 1 - np.sqrt(sse/4)
