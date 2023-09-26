import numpy as np
from dataclasses import dataclass

def State(state):
  return state | {'index': 0, 'scope': ()}

def model(apply):
  def model(state, x):

    state1 = state | {'index': state['index'] + 1}
    state2, x = apply(state | {'scope': state['scope'] + (state['index'],)}, x)
    
    return state1 | state2, x
  return model

@model
def parameter(state, x):
  state = {state['scope']: x} | state

  return state, state[state['scope']]
