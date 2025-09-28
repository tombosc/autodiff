import autodiff as ad
import numpy as np

graph_viz = ad.graph.Node.visualize

def f(v):
    return ad.ops.vec_sum(v)

def g(A, v):
    v2 = ad.ops.mat_vec(A, v)
    return ad.ops.vec_sum(v2)



if __name__ == "__main__":

    ones = ad.numpy.ones((3,))
    print(f(ones))

    graph = ad.graph.build_graph(f, [ones])
    graph_viz(graph)
    compiled_f = ad.graph.get_forward_fn(graph)
    print(compiled_f(v=ones))

    print("Harder")
    M = ad.numpy.Array(
        np.asarray([[1, 0, 0.5], [0, 1, 0], [0, 1, 1], [0.2, 1, 1]])
    )
    v = ad.numpy.Array(np.asarray([2., 1., 0.]))
    # print("Matrix", M)
    # print("Transpose", ad.ops.mat_transpose(M))
    # print(g(M, ones))
    graph = ad.graph.build_graph(g, [M, ones])
    # graph_viz(graph)
    compiled_f = ad.graph.get_forward_fn(graph)
    print("result", compiled_f(A=M, v=ones))
    grad_g = ad.graph.grad(g, [M, ones])
    one = ad.numpy.Array(np.ones((1,)))
    ones_col = ad.numpy.Array(np.ones((4,)))
    for name, grad in grad_g.items(): 
        print(f"Gradient wrt. {name}")
        grad_f = lambda ones: grad
        new_graph = ad.graph.build_graph(grad_f, [one])
        # graph_viz(new_graph)
        fn = ad.graph.get_forward_fn(new_graph)
        val_grad = fn(ones=one, ones_for_grad=ones_col, v=v, A=M)
        print(f"Value of grad {val_grad}")
        breakpoint()
