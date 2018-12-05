import lasagne.layers

class Pad(lasagne.layers.PadLayer):
    """
    This layer performs unpooling over the last two dimensions of a 4D tensor.
    """

    # for convenience
    isdense, isbatchnorm, isconv, isdropout, isactivation, ismaxpool = (False,) * 6

    def __init__(self, incoming_layer, width = 0, val = 0, batch_ndim = 2, name=None,
                 **kwargs):

        lasagne.layers.PadLayer.__init__(self, incoming_layer,
                                               width=width, val=val, batch_ndim = batch_ndim,
                                               name=name, **kwargs)

        self.inp_ndim = 4
