import autodiff as ad
import numpy as np
import pytest


graph_viz = ad.graph.Node.visualize


def AB(A, B):
    return ad.ops.mat_mat_mul(A, B)


def test_mat_mat_mul_forward():
    A = ad.numpy.Array(np.asarray([[1, 0, 0.5], [0, 1, 0], [0, 1, 1]]))
    B = ad.numpy.Array(np.asarray([[1, 0, 0.], [0, 1, 0], [0, 1, 1]]))
    test_graph = ad.graph.build_graph(AB, [A, B])
    graph_viz(test_graph)
    compiled_f = ad.graph.get_forward_fn(test_graph)
    result = compiled_f(A=A, B=B)
    assert(np.array_equal(
        result.tensor, 
        np.asarray([[1, 0.5, 0.5], [0, 1, 0], [0, 2, 1]]),
    ))


def quad(A, b, x):
    Ax = ad.ops.mat_mat_mul(A, x)
    xTAx = ad.ops.mat_mat_mul(ad.ops.mat_transpose(x), Ax)
    bTx = ad.ops.mat_mat_mul(ad.ops.mat_transpose(b), x)
    return ad.ops.mat_mat_sum(xTAx, bTx)
    # return ad.ops.mat_mat_sum(bTx, xTAx)

@pytest.fixture
def problem():
    """Fixture providing the quadratic problem parameters and solution."""
    A_val = np.asarray([[20, 0.], [0, 1]])
    b_val = np.asarray([[0.], [1.]])
    x_init = np.asarray([[1.], [1.]])
    sol = np.asarray([[0., -0.5]]).T
    return {
        "A": ad.numpy.Array(A_val),
        "b": ad.numpy.Array(b_val),
        "x_init": ad.numpy.Array(x_init),
        "solution": sol
    }


def test_quad_opt_GD(problem):
    A, b, x = problem["A"], problem["b"], problem["x_init"]
    solution = problem["solution"]
    A = ad.numpy.Array(np.asarray([[20, 0.], [0, 1]]))
    b = ad.numpy.Array(np.asarray([[0.], [1.]]))
    x = ad.numpy.Array(np.asarray([[1.], [1.]]))
    graph_loss = ad.graph.build_graph(quad, [A, b, x])
    grad, adj_name = ad.graph.grad(graph_loss, [A, b, x])
    compiled_grad_f_x = ad.graph.get_forward_fn(grad["x"])
    ones = ad.numpy.Array(np.ones((1, 1)))
    cst = {adj_name: ones}
    lr = 0.03
    for i in range(1000):
        print("x iteration", x.tensor)
        grad_x = compiled_grad_f_x(A=A, b=b, x=x, **cst)
        x.tensor = x.tensor - lr * grad_x.tensor
    assert(np.allclose(x.tensor, solution, rtol=1e-5))

def test_quad_opt_newton1(problem):
    """ Test Newton's method by computing full Hessian.
    """
    A, b, x = problem["A"], problem["b"], problem["x_init"]
    solution = problem["solution"]
    # TODO test Newton's method by computing grad x vector dot product
    graph_loss = ad.graph.build_graph(quad, [A, b, x])
    loss_fn = ad.graph.get_forward_fn(graph_loss)
    loss_val = loss_fn(A=A, b=b, x=x)
    assert(np.array_equal(loss_val.tensor, np.asarray([[22.]])))
    graphs_grad, adj_name = ad.graph.grad(graph_loss, [A, b, x])
    compiled_grad_f_x = ad.graph.get_forward_fn(graphs_grad["x"])
    ones = ad.numpy.Array(np.ones((1, 1)))
    print("First adjoint 1s name", adj_name)
    cst1 = {adj_name: ones}
    grad_x = compiled_grad_f_x(A=A, b=b, x=x, **cst1)
    graph_H, adj_name = ad.graph.grad(graphs_grad["x"], [A, b, x])
    print("2nd adjoint 1s name", adj_name)
    graph_H = graph_H["x"]
    H_fn = ad.graph.get_forward_fn(graph_H, verbose_level=1)
    # build Hessian
    H_to_stack = []
    for adj in np.eye(2):
        cst1.update({
            adj_name: ad.numpy.Array(adj[:, np.newaxis])
        })
        H_val = H_fn(A=A, B=b, x=x, **cst1)
        H_to_stack.append(H_val.tensor)
    H = np.hstack(H_to_stack)
    assert(np.array_equal(H, (A.tensor + A.tensor.T)))
    H_inv = np.linalg.inv(H)
    x.tensor = x.tensor - np.matmul(H_inv, grad_x.tensor)
    print(x.tensor)
    assert(np.allclose(x.tensor, solution, rtol=1e-5))
