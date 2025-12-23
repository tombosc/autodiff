import autodiff as ad
import numpy as np


def vec_sum(v):
    if isinstance(v, ad.graph.Node):
        assert len(v.shape) == 1, "invalid shape"
        new_node = ad.graph.Node(op=vec_sum, shape=(0,), children=[v])
        return new_node
    elif isinstance(v, ad.numpy.Array):
        return ad.numpy.Array(v.tensor.sum())
    else:
        raise ValueError("Unrecognized value")


def vec_vec_mul(u, v):
    if isinstance(u, ad.graph.Node) and isinstance(v, ad.graph.Node):
        assert len(u.shape) == 1, "invalid shape"
        assert len(v.shape) == 1, "invalid shape"
        return ad.graph.Node(op=vec_vec_mul, shape=(0,), children=[u, v])
    elif isinstance(u, ad.numpy.Array) and isinstance(v, ad.numpy.Array):
        return ad.numpy.Array(np.dot(u.tensor, v.tensor))
    else:
        raise ValueError("Unrecognized value")


def vec_vec_sum(u, v):
    if all_nodes([u, v]):
        assert len(u.shape) == 1, "invalid shape"
        assert len(v.shape) == 1, "invalid shape"
        return ad.graph.Node(
            op=vec_vec_sum, 
            shape=(0,),
            children=[u, v],
        )
    elif all_arrays([u, v]):
        return ad.numpy.Array(u.tensor + v.tensor)
    else:
        raise ValueError("Unrecognized value")


def mat_mat_sum(u, v):
    if all_nodes([u, v]):
        assert len(u.shape) == 2, "invalid shape"
        assert len(v.shape) == 2, "invalid shape"
        return ad.graph.Node(
            op=mat_mat_sum, 
            shape=(1, 1),
            children=[u, v],
            name="sum",
        )
    elif all_arrays([u, v]):
        return ad.numpy.Array(u.tensor + v.tensor)
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
        raise ValueError("Unrecognized value")

def mat_mat_mul(A, B):
    if all_nodes([A, B]):
        assert len(A.shape) == 2, "invalid shape"
        assert len(B.shape) == 2, "invalid shape"
        # TODO: can we assert A.shape[1] == B.shape[0]?
        return ad.graph.Node(
            op=mat_mat_mul, 
            shape=(A.shape[0],B.shape[1]),
            children=[A, B],
        )
    elif all_arrays([A, B]):
        return ad.numpy.Array(A.tensor @ B.tensor)
    else:
        raise ValueError("Unrecognized value")

def scal_vec_mul(a, v):
    if all_nodes([a, v]):
        print("scal_vec_mul assert shapes", a.shape, v.shape)
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


def vec_sum_VJP(u, v):
    if isinstance(v, ad.numpy.Array):
        one = ad.numpy.ones((v.tensor.shape[1],))
    elif isinstance(v, ad.graph.Node):
        one = ad.graph.Node(op=None, shape=(1,), children=[], name="ones_for_grad")
    # Use ops here! Not numpy directly!
    return (scal_vec_mul(one, u),)


def vec_vec_sum_VJP(u, v, w):
    u_col = mat_transpose(vec_to_row_mat(u))
    return (u_col, u_col)


def mat_mat_sum_VJP(C, A, B):
    return (C, C)


def mat_vec_VJP(u, M, v):
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


def vec_vec_mul_VJP(u, v, w):
    return (
        scal_vec_mul(u, w),
        scal_vec_mul(u, v),
    )


def scal_vec_mul_VJP(u, a, v):
    return (
        vec_vec_mul(u, v),  
        scal_vec_mul(a, u),
    )


def mat_mat_mul_VJP(C, A, B):
    return (
        mat_mat_mul(C, mat_transpose(B)),
        mat_mat_mul(mat_transpose(A), C),
    )

def mat_transpose_VJP(C, A):
    return (
        mat_transpose(C),
    )

VJP_map = {
    mat_mat_mul: mat_mat_mul_VJP,
    # vec_vec_sum: vec_vec_sum_VJP,
    mat_mat_sum: mat_mat_sum_VJP,
    # vec_vec_mul: vec_vec_mul_VJP,
    # vec_sum: vec_sum_VJP,
    # mat_vec: mat_vec_VJP,
    mat_transpose: mat_transpose_VJP,
    # scal_vec_mul: scal_vec_mul_VJP,
    # TODO rest
}
