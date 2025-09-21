Goal:

Have a reverse-mode autodiff tool that can compute gradients for simple functions.

Inspired by Jax, so that I can then play with more complex variations on autodiff.

Jaxpr == Graph
ClosedJaxpr will be a tuple (Graph, Assignment) where all the leaves are assigned variables in the `Assignement`.

IR will not be optimized.
To run the forward, we will go from root (=result) to leaves, assign values to those, and then return by applying.
