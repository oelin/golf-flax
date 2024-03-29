# golf-flax

A golfed version of the Flax module API. Implementd in 20 lines. Fully compatable with JAX. Supports module inlining.

## Implementation

```python
def State(state):
  return state | {'index': 0, 'scope': ()}

def model(apply):
  def model(state, x):

    state1 = state | {'index': state['index'] + 1}
    state2, x = apply(state | {'scope': state['scope'] + (state['index'],)}, x)
    
    return state2 | state1, x
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
    state, x = nn.Linear(2, 2)(state, x)
    state, x = nn.Linear(2, 1)(state, x)

    return state, x

# Initialization.

state, x = MLP(State({}), jnp.ones((2, 2)))

# Inference.

x, state = MLP(State(state), jnp.ones((2, 2)))
```
