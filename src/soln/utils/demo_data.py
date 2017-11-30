from soln.utils.data_utils import to_vector

def get_demo_data(n_inputs, n_outputs):
    out = []
    for i in xrange(n_outputs):
        out.append([(to_vector(i, size=n_inputs), to_vector(i, size = n_outputs))] * 10)
    return out

