import autodiff as ad
import numpy as np
from inspect import getmembers, isfunction, signature
from functools import partial


class Node():
    global_inc = 0

    def __init__(self, op, shape, children, name=None):
        self.op = op
        self.shape = shape
        self.children = children
        self.name = name
        if name is None:
            self.name = "aaaa_" + str(self.global_inc)
            self.global_inc += 1

    @staticmethod
    def visualize(node, indent=0):
        print("  " * indent + f"{node.name} (op={node.op}, shape={node.shape})")
        for child in node.children:
            Node.visualize(child, indent + 1)

    # TODO implement ADD


def build_graph(f, args):
        """ Takes a function f and builds a graph.
        """
        params_f = signature(f).parameters
        assert len(args) == len(params_f)
        def leaf(shape, name):
            return Node(op=None, shape=shape, children=[], name=name)
        tracers = [leaf(a.shape, p) for a, p in zip(args, params_f)]
        G = f(*tracers)
        return G

def get_forward_fn(graph):
    def exec_numpy(node, **kwargs):
        """ v is a list of tensors.
        """
        if len(node.children) == 0 and node.op == None:
            # leaf!
            # assign value
            node._val = kwargs[node.name]
            return
        values = []
        for child in node.children:
            exec_numpy(child, **kwargs)
            values.append(child._val)
        assert node.op is not None, "Op is None but node not leaf?"
        node._val = node.op(*values)
        return node._val
    return partial(exec_numpy, node=graph)
