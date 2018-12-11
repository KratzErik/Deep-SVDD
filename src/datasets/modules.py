from config import Configuration as Cfg
from lasagne.init import GlorotUniform, Constant


def addConvModule(nnet, num_filters, filter_size, pad='valid', W_init=None, bias=True, use_maxpool = True, pool_size=(2,2),
                  use_batch_norm=False, dropout=False, p_dropout=0.5, upscale=False, stride = (1,1)):
    """
    add a convolutional module (convolutional layer + (leaky) ReLU + MaxPool) to the network  
    """

    if W_init is None:
        W = GlorotUniform(gain=(2/(1+0.01**2)) ** 0.5)  # gain adjusted for leaky ReLU with alpha=0.01
    else:
        W = W_init

    if bias is True:
        b = Constant(0.)
    else:
        b = None

    # build module
    if dropout:
        nnet.addDropoutLayer(p=p_dropout)

    nnet.addConvLayer(use_batch_norm=use_batch_norm,
                      num_filters=num_filters,
                      filter_size=filter_size,
                      pad=pad,
                      W=W,
                      b=b,
                      stride = stride)

    if Cfg.leaky_relu:
        nnet.addLeakyReLU()
    else:
        nnet.addReLU()

    if upscale:
        nnet.addUpscale(scale_factor=pool_size)
    elif use_maxpool:
        nnet.addMaxPool(pool_size=pool_size)

def addConvTransposeModule(nnet, num_filters, filter_size, W_init=None, bias=True, use_maxpool = False, pool_size=(2,2),
                  use_batch_norm=False, dropout=False, p_dropout=0.5, upscale=False, stride = (1,1), crop=0, outpad=0, inpad=0, output_size = None):
    """
    add a convolutional module (convolutional layer + (leaky) ReLU + MaxPool) to the network  
    """

    if W_init is None:
        W = GlorotUniform(gain=(2/(1+0.01**2)) ** 0.5)  # gain adjusted for leaky ReLU with alpha=0.01
    else:
        W = W_init

    if bias is True:
        b = Constant(0.)
    else:
        b = None

    # build module
    if dropout:
        nnet.addDropoutLayer(p=p_dropout)

    if inpad > 0:
        print("Adding assymmetric input padding: %d"%inpad)
        nnet.addPadLayer(width=((inpad,0),(0, inpad)))

    nnet.addConvTransposeLayer(use_batch_norm=use_batch_norm,
                      num_filters=num_filters,
                      filter_size=filter_size,
                      W=W,
                      b=b,
                      stride=stride, crop=crop, output_size = output_size)

    if outpad > 0:
        print("Adding assymmetric output padding: %d"%outpad)
        nnet.addPadLayer(width=((outpad,0),(0,outpad)))

    if Cfg.leaky_relu:
        nnet.addLeakyReLU()
    else:
        nnet.addReLU()

    if upscale:
        nnet.addUpscale(scale_factor=pool_size)
    elif use_maxpool:
        nnet.addMaxPool(pool_size=pool_size)
