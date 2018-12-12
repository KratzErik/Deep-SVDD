import cPickle as pickle
from config import Configuration as Cfg
from theano import shared

def dump_weights(nnet, filename=None, pretrain=False, epoch = 0):

    if filename is None:
        filename = nnet.pickle_filename

    weight_dict = dict()

    for layer in nnet.trainable_layers:
        weight_dict[layer.name + "_w"] = layer.W.get_value()
        if layer.b is not None:
            weight_dict[layer.name + "_b"] = layer.b.get_value()

    for layer in nnet.all_layers:
        if layer.isbatchnorm:
            weight_dict[layer.name + "_beta"] = layer.beta.get_value()
            weight_dict[layer.name + "_gamma"] = layer.gamma.get_value()
            weight_dict[layer.name + "_mean"] = layer.mean.get_value()
            weight_dict[layer.name + "_inv_std"] = layer.inv_std.get_value()

    if Cfg.svdd_loss and not pretrain:
            weight_dict["R"] = nnet.Rvar.get_value()
            weight_dict["c"] = nnet.cvar.get_value()

    if "checkpoint" in filename:
        print("Saving checkpoint at epoch ", epoch)
        weight_dict["checkpoint_epoch"] = epoch

    with open(filename, 'wb') as f:
        pickle.dump(weight_dict, f)

    print("Parameters saved in %s" % filename)


def load_weights(nnet, filename=None):

    if filename is None:
        filename = nnet.pickle_filename

    print("Loading weights from %s"%filename)

    with open(filename, 'rb') as f:
        weight_dict = pickle.load(f)

    for layer in nnet.trainable_layers:
        layer.W.set_value(weight_dict[layer.name + "_w"])
        if layer.b is not None:
            layer.b.set_value(weight_dict[layer.name + "_b"])
    print("\tLoaded trainable layers")
    for layer in nnet.all_layers:
        if layer.isbatchnorm:
            layer.beta.set_value(weight_dict[layer.name + "_beta"])
            layer.gamma.set_value(weight_dict[layer.name + "_gamma"])
            layer.mean.set_value(weight_dict[layer.name + "_mean"])
            layer.inv_std.set_value(weight_dict[layer.name + "_inv_std"])
    print("\tLoaded all layers")

    if Cfg.svdd_loss:
        if "R" in weight_dict:
            nnet.R_init = weight_dict["R"]
            #print("\tSet R value to saved: %.f"%nnet.R_init)
        else:
            print("\tNo R value saved")
        if "c" in weight_dict:
            nnet.cvar = shared(weight_dict["c"])
            #print("\tSet c value to saved: %.f"%nnet.cvar)
        else:
            print("\tNo c value saved")

    if "ae_checkpoint" in filename:
        nnet.ae_checkpoint_epoch = weight_dict["checkpoint_epoch"]
        print("AE checkpoint at ", nnet.ae_checkpoint_epoch)
    elif "checkpoint" in filename:
        nnet.checkpoint_epoch = weight_dict["checkpoint_epoch"]
        print("Checkpoint at ", nnet.checkpoint_epoch)
    print("Parameters loaded in network")


def dump_svm(model, filename=None):

    with open(filename, 'wb') as f:
        pickle.dump(model.svm, f)

    print("Model saved in %s" % filename)


def load_svm(model, filename=None):

    print("Loading model...")

    with open(filename, 'rb') as f:
        model.svm = pickle.load(f)

    print("Model loaded.")


def dump_kde(model, filename=None):

    with open(filename, 'wb') as f:
        pickle.dump(model.kde, f)

    print("Model saved in %s" % filename)


def load_kde(model, filename=None):

    print("Loading model...")

    with open(filename, 'rb') as f:
        model.kde = pickle.load(f)

    print("Model loaded.")

def dump_isoForest(model, filename=None):

    with open(filename, 'wb') as f:
        pickle.dump(model.isoForest, f)

    print("Model saved in %s" % filename)


def load_isoForest(model, filename=None):

    print("Loading model...")

    with open(filename, 'rb') as f:
        model.isoForest = pickle.load(f)

    print("Model loaded.")
