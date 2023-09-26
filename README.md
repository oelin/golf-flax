# golf-flax

A golfed version of the Flax module API. Implementd in 20 lines. Fully compatable with JAX. Supports module inlining.

## Implementation

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

## Usage

```python
import nn

@model
def MLP(state, x):
    state, x = nn.Linear(2, 2)(ctx, x)
    state, x = nn.Linear(2, 1)(ctx, x)

    return state, x

## Future

Hoping to make this even more compact by merging `x` and `state`.

# Initialization.

x, state = MLP(State({}), jnp.ones((2, 2)))

# Inference.

x, state = MLP(State(state), jnp.ones((2, 2)))
```
