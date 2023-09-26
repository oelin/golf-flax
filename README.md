# golf-flax

A golfed version of the Flax module API. Implementd in 20 lines. Fully compatable with JAX. Supports module inlining.

```python
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
```

Usage:

```python
import nn

@model
def MLP(state, x):
    state, x = nn.Linear(2, 2)(ctx, x)
    state, x = nn.Linear(2, 1)(ctx, x)

    return state, x
```
