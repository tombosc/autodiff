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

def matmul(M, v):
    if isinstance(v, ad.graph.Node) and isinstance(M, ad.graph.Node):
        assert len(M.shape) == 2, "invalid shape"
        assert len(v.shape) == 1, "invalid shape"
        return ad.graph.Node(
            op=matmul, 
            shape=(1,),
            children=[M, v],
        )
    elif isinstance(M, ad.numpy.Array) and isinstance(v, ad.numpy.Array):
        return ad.numpy.Array(
            np.matmul(M.tensor, v.tensor)
        )
    else:
        raise ValueError("Unrecognized value")


