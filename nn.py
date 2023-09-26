from flax import model 


def Linear(n, m):
  def linear(ctx, x):
    ctx, w = parameter(ctx, 'w', np.random.randn(m, n))
    ctx, b = parameter(ctx, 'b', np.random.randn(m))

    return ctx, x @ w.T + b
  return model(linear)
