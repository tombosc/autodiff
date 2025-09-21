import autodiff as ad
import numpy as np

def f(v):
    return ad.ops.vec_sum(v)

def g(A, v):
    v2 = ad.ops.matmul(A, v)
    return ad.ops.vec_sum(v2)# + 0.1


if __name__ == "__main__":

    ones = ad.numpy.ones((3,))
    print(f(ones))

    graph = ad.graph.build_graph(f, [ones])
    ad.graph.Node.visualize(graph)
    compiled_f = ad.graph.get_forward_fn(graph)
    print(compiled_f(v=ones))

    print("Harder")
    M = ad.numpy.Array(
        np.asarray([[1, 0, 0.5], [0, 1, 0], [0, 1, 1]])
    )
    print(g(M, ones))
    graph = ad.graph.build_graph(g, [M, ones])
    ad.graph.Node.visualize(graph)
    compiled_f = ad.graph.get_forward_fn(graph)
    print(compiled_f(A=M, v=ones))
