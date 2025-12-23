import autodiff as ad
import numpy as np
from inspect import getmembers, isfunction, signature
from functools import partial
import uuid


class Dimension():
    global_inc = 0

    def __init__(self):
        self.name = "d_" + str(self.global_inc)
        Dimension.global_inc += 1

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name


class Node():
    global_inc = 0

    def __init__(self, op, shape, children, name=None):
        self.op = op
        self.shape = []
        for dim in shape:
            if isinstance(dim, int):
                self.shape.append(Dimension())
            elif isinstance(dim, Dimension):
                self.shape.append(dim)
            else:
                raise ValueError("Unrecognized shape")
        self.children = children
        self.name = name
        if name is None:
            self.name = "node_" + str(self.global_inc)
            Node.global_inc += 1

    @staticmethod
    def visualize(node, indent=0):
        shape_str = "(" + ", ".join([str(d) for d in node.shape]) + ")"
        print("  " * indent + f"{node.name} (op={node.op}, shape={shape_str})")
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


def get_forward_fn(graph, verbose_level=0):
    def exec_numpy(node, **kwargs):
        """ v is a list of tensors.
        """
        if len(node.children) == 0 and node.op == None:  # leaf
            node._val = kwargs[node.name]
            return
        values = []
        for child in node.children:
            if verbose_level >= 1:
                print(f"Computing child {child.name}")
            exec_numpy(child, **kwargs)
            values.append(child._val)
        assert node.op is not None, "Op is None but node not leaf?"
        node._val = node.op(*values)
        return node._val
    return partial(exec_numpy, node=graph)


def grad(graph_root, params, verbose_level=0):
    """ Backward-mode diff
    """
    print("EXEC grad")
    VJP_map = ad.ops.VJP_map
    # starting from graph_root node (=loss), make a function that composes VJP
    # one = ad.numpy.ones((1,))
    n_dim_root = len(graph_root.shape)
    short_id = uuid.uuid4().hex[:6]
    name_ones = f"ones_{short_id}"
    shape_ones = (0,)*n_dim_root
    one = Node(op=None, shape=shape_ones, children=[], name=name_ones)
    node_list = [(graph_root, one)]
    name_to_grad = {}
    while len(node_list):
        cur_node, cur_u = node_list.pop()
        if cur_node.op != None:
            VJP = VJP_map[cur_node.op]
            # VJP is a function of cur_u and all the other arguments of the function
            children = [c for c in cur_node.children]
            current_us = VJP(cur_u, *children)
            assert(len(current_us) == len(children))
            for child, u in zip(children, current_us):
                node_list.append((child, u))
        else:
            # leaf, store gradients!
            name = cur_node.name
            print(f"Leaf... storing gradient {name}!")
            # Node.visualize(cur_u)
            cur_grad = name_to_grad.get(name, None)
            if cur_grad is not None:
                name_to_grad[name] = ad.ops.mat_mat_sum(cur_grad, cur_u)
            else:
                name_to_grad[name] = cur_u
    return name_to_grad, name_ones
