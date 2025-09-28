Educational autodiff tool inspired by jax, that can compute gradients of simple functions.

Goals:

- How do Jaxpr and pytrees interact (for now, no such thing, only a computation graph)?
- Can it compute second-order derivatives?
- How to deal with constants better (now I need to pass them)
- How to do broadcasting to make the code much simpler (ie avoid all those ops that convert from vec to row / vector matrices)
