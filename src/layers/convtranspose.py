import lasagne.layers
import theano.tensor as T


class ConvTransposeLayer(lasagne.layers.TransposedConv2DLayer):

    # for convenience
    isdense, isbatchnorm, isdropout, ismaxpool, isactivation = (False,) * 5
    isconv = True

    def __init__(self, incoming_layer, num_filters, filter_size, stride=(1, 1), crop = 0,
                 W=lasagne.init.GlorotUniform(gain='relu'),
                 b=lasagne.init.Constant(0.), flip_filters=True, name=None, output_size = None):
        if output_size is None:
            lasagne.layers.TransposedConv2DLayer.__init__(self, incoming_layer, num_filters,
                                                filter_size, name=name,
                                                stride=stride, crop='full',
                                                untie_biases=False, W=W, b=b,
                                                nonlinearity=None,
                                                flip_filters=flip_filters)
        else:
            lasagne.layers.TransposedConv2DLayer.__init__(self, incoming_layer, num_filters,
                                                filter_size, name=name,
                                                stride=stride, crop='full',
                                                untie_biases=False, W=W, b=b,
                                                nonlinearity=None,
                                                flip_filters=flip_filters, output_size=output_size)

        self.inp_ndim = 4
        self.use_dc = False
