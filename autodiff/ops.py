import autodiff as ad
import numpy as np


def vec_sum(v):
    if isinstance(v, ad.graph.Node):
        assert len(v.shape) == 1, "invalid shape"
        new_node = ad.graph.Node(
            op=vec_sum, 
            shape=(0,),
            children=[v],
        )
        return new_node
    elif isinstance(v, ad.numpy.Array):
        return ad.numpy.Array(v.tensor.sum())
    else:
        raise ValueError("Unrecognized value")


def vec_vec_mul(u, v):
    if isinstance(u, ad.graph.Node) and isinstance(v, ad.graph.Node):
        assert len(u.shape) == 1, "invalid shape"
        assert len(v.shape) == 1, "invalid shape"
        return ad.graph.Node(
            op=vec_vec_mul, 
            shape=(0,),
            children=[u, v],
        )
    elif isinstance(u, ad.numpy.Array) and isinstance(v, ad.numpy.Array):
        return ad.numpy.Array(
            np.dot(u.tensor, v.tensor)
        )
    else:
        raise ValueError("Unrecognized value")


def mat_transpose(M):
    if isinstance(M, ad.graph.Node):
        assert len(M.shape) == 2, "invalid shape"
        return ad.graph.Node(
            op=mat_transpose, 
            shape=(1,1),
            children=[M],
        )
    elif isinstance(M, ad.numpy.Array):
        return ad.numpy.Array(np.transpose(M.tensor))
    else:
        raise ValueError("Unrecognized value")

def vec_transpose(v):
    if isinstance(v, ad.graph.Node):
        assert len(v.shape) == 1, "invalid shape"
        return ad.graph.Node(
            op=vec_transpose, 
            shape=(1,),
            children=[v],
        )
    elif isinstance(v, ad.numpy.Array):
        return ad.numpy.Array(np.transpose(v.tensor))
    else:
        raise ValueError("Unrecognized value")

def allinstance(type_, args):
    return all([isinstance(arg, type_) for arg in args])

def all_nodes(args):
    return allinstance(ad.graph.Node, args)

def all_arrays(args):
    return allinstance(ad.numpy.Array, args)

def mat_vec(M, v):
    if all_nodes([M, v]):
        assert len(M.shape) == 2, "invalid shape"
        assert len(v.shape) == 1, "invalid shape"
        return ad.graph.Node(
            op=mat_vec, 
            shape=(1,),
            children=[M, v],
        )
    elif all_arrays([M, v]):
        return ad.numpy.Array(
            np.dot(M.tensor, v.tensor)
        )
    else:
        breakpoint()
        raise ValueError("Unrecognized value")

def mat_mat_mul(A, B):
    if all_nodes([A, B]):
        assert len(A.shape) == 2, "invalid shape"
        assert len(B.shape) == 2, "invalid shape"
        return ad.graph.Node(
            op=mat_vec, 
            shape=(1,1),
            children=[A, B],
        )
    elif all_arrays([A, B]):
        return ad.numpy.Array(A.tensor @ B)
    else:
        raise ValueError("Unrecognized value")

def scal_vec_mul(a, v):
    if all_nodes([a, v]):
        # TODO implement scalar...
        assert len(a.shape) == 1, "invalid shape"
        assert len(v.shape) == 1, "invalid shape"
        return ad.graph.Node(
            op=scal_vec_mul, 
            shape=(1,),
            children=[a, v],
        )

    elif all_arrays([a, v]):
        return ad.numpy.Array(a.tensor * v.tensor)
    else:
        raise ValueError("Unrecognized value")

def vec_to_row_mat(v):
    if isinstance(v, ad.graph.Node):
        assert len(v.shape) == 1, "invalid shape"
        return ad.graph.Node(
            op=vec_to_row_mat, 
            shape=(1,1),
            children=[v],
        )
    elif isinstance(v, ad.numpy.Array):
        return ad.numpy.Array(
            v.tensor.reshape((1, -1))
        )
    else:
        raise ValueError("Unrecognized value")

# it's key to use ops here, and not numpy directly
def vec_sum_VJP(u, v):
    if isinstance(v, ad.numpy.Array):
        one = ad.numpy.ones((v.tensor.shape[1],))
    elif isinstance(v, ad.graph.Node):
        one = ad.graph.Node(op=None, shape=(1,), children=[], name="ones_for_grad")
    return (scal_vec_mul(one, u),)

def mat_vec_VJP(u, M, v):
    # if isinstance(M, ad.numpy.Array):
    #     ones = ad.numpy.ones((1,))
    # elif isinstance(M, ad.graph.Node):
    #     ones = ad.graph.Node(op=None, shape=(0,), children=[], name="ones")

    # ȳ corresponds to the adjoint, argument u of the python function
    # ȳᵀdy = ȳᵀ(Mdv + dMv)
    #      = (Mdv)ᵀȳ + Tr(ȳᵀdMv)
    #      = dvᵀ(Mᵀȳ) + Tr(vȳᵀdM)
    #      = dvᵀ(Mᵀȳ) + Tr(dMᵀȳvᵀ)
    # ⇒ VJP for v is Mᵀȳ, for M: ȳvᵀ
    u_col = mat_transpose(vec_to_row_mat(u))
    return (
        mat_mat_mul(u_col, vec_to_row_mat(v)),  # VJP of M
        vec_to_row_mat(mat_vec(mat_transpose(M), u)),  # VJP of v
    )

def scal_vec_mul_VJP(u, a, v):
    return (
        scal_vec_mul(u, v),  
        mat_transpose(scal_vec_mul(a, u))  
    )


VJP_map = {
    vec_sum: vec_sum_VJP,
    mat_vec: mat_vec_VJP,
    scal_vec_mul: scal_vec_mul_VJP,
    # TODO rest
}
