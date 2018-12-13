import numpy as np
import theano
from pathlib import Path
from datasets import loadbdd100k


class Configuration(object):
    only_test = bool(0)
    export_results = bool(0)
    dataset = "prosivic"

    floatX = np.float32
    seed = 0

    n_pretrain_epochs = 2000
    pretrain_learning_rate = 0.001 # standard DSVDD was 0.0001
    plot_filters = bool(0)
    plot_most_out_and_norm = bool(0)
    debug_architecture_layers = bool(0) # if true, layer output shapes will be printed at compilation for both autoencoder and CNN
    use_checkpoint = True
    checkpoint_interval = 10

    if dataset == 'dreyeve':
        # Dr(eye)ve parameters
        train_folder = "../../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/train/"
        val_folder = "../../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/val/"
        test_in_folder = "../../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/test/in/"
        test_out_folder = "../../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/test/out/"
        #train_folder = "../../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/train/"
        #val_folder = "../../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/val/"
        #test_in_folder = "../../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/test/in/"
        #test_out_folder = "../../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/test/out/"
        data_div = 1
        n_train = 6000 // data_div
        n_val = 600 // data_div
        n_test = 1200 // data_div
        n_test_in = 600 // data_div
        image_height = 256
        image_width = 256
        channels = 3
        rep_dim = 512
    #    architecture = 1
        architecture = '0_6_1_16_512_5_2_2'
        bias = True
        n_dict_learn = min(500,n_train)

    elif dataset == 'prosivic':
        # Pro-SiVIC parameters
        train_folder = "../../weather_detection_data/prosivic/train/"
        val_folder = "../../weather_detection_data/prosivic/val/"
        test_in_folder = "../../weather_detection_data/prosivic/test/in/"
        test_out_folder = "../../weather_detection_data/prosivic/test/out/foggy/"
        data_div = 1
        n_train = 7000 // data_div
        n_val = 1400 // data_div
        n_test = 784*2 // data_div
        n_test_in = 784 // data_div
        image_height = 256
        image_width = 256
        channels = 3
        rep_dim = 512
    #    architecture = 1
        architecture = '0_5_1_16_512_5_2_2'
        bias = True
        n_dict_learn = min(500,n_train)


    # BDD100K parameters, currently not used
    bdd100k_use_file_lists = True
    bdd100k_file_list_normal = 'clear_overcast_partlycloudy_highway_daytime.txt'
    bdd100k_file_list_outlier = 'rainy_foggy_snowy_highway_anytime.txt'
    bdd100k_attributes_normal = [["weather", ["clear","partly cloudy", "overcast"]],["scene", "highway"],["timeofday", "daytime"]]
    bdd100k_attributes_outlier = [["scene", "highway"],["weather", ["rainy", "snowy", "foggy"]],["timeofday",["daytime","dawn/dusk","night"]]]
    bdd100k_img_folder = Path("/data/bdd100k/images/train_and_val_256by256")
    bdd100k_norm_file = "/data/bdd100k/namelists/clear_or_partly_cloudy_or_overcast_and_highway_and_daytime.txt"
    bdd100k_norm_filenames = loadbdd100k.get_namelist_from_file(bdd100k_norm_file)
    bdd100k_out_file = "/data/bdd100k/namelists/rainy_or_snowy_or_foggy_and_highway_and_daytime_or_dawndusk_or_night.txt"
    bdd100k_out_filenames = loadbdd100k.get_namelist_from_file(bdd100k_out_file)
    bdd100k_norm_spec = [["weather", ["clear","partly cloudy", "overcast"]],["scene", "highway"],["timeofday", "daytime"]]
    bdd100k_out_spec = [["weather", ["rainy", "snowy", "foggy"]],["scene", "highway"],["timeofday",["daytime","dawn/dusk","night"]]]
    bdd100k_n_train = 2048
    bdd100k_n_val = 1024
    bdd100k_n_test = 1024
    bdd100k_out_frac = 0.2
    bdd100k_image_height = 256
    bdd100k_image_width = 256
    bdd100k_channels = 3
    bdd100k_save_name_lists=False
    bdd100k_labels_file = "/data/bdd100k/labels/bdd100k_labels_images_train_and_val.json"
    bdd100k_get_norm_and_out_sets = False
    bdd100k_shuffle=False
    bdd100k_rep_dim = 512
    bdd100k_architecture = 1
    bdd100k_bias = True
    bdd100k_n_dict_learn = min(500,bdd100k_n_train)

    if dataset == "bdd100k":
        n_train = bdd100k_n_train
        n_val = bdd100k_n_val
        n_test = bdd100k_n_test
        n_test_in = int(n_test*(1-bdd100k_out_frac))
    # Final Layer
    softmax_loss = False
    svdd_loss = False
    reconstruction_loss = False

    # Optimization
    batch_size = 64
    learning_rate = theano.shared(floatX(1e-4), name="learning rate")
    lr_decay = False
    lr_decay_after_epoch = 10
    lr_drop = False  # separate into "region search" and "fine-tuning" stages
    lr_drop_factor = 10
    lr_drop_in_epoch = 50
    momentum = theano.shared(floatX(0.9), name="momentum")
    rho = theano.shared(floatX(0.9), name="rho")
    use_batch_norm = True  # apply batch normalization

    eps = floatX(1e-8)

    # Network architecture
    leaky_relu = False
    dropout = False
    dropout_architecture = False

    # Pre-training and autoencoder configuration
    weight_dict_init = False
    pretrain = False
    ae_loss = "l2"
    ae_lr_drop = True  # separate into "region search" and "fine-tuning" stages
    ae_lr_drop_factor = 10
    ae_lr_drop_in_epoch = int(n_pretrain_epochs * 2/3)
    ae_weight_decay = False
    ae_C = theano.shared(floatX(1e3), name="ae_C")

    # Regularization
    weight_decay = True
    C = theano.shared(floatX(1e3), name="C")
    reconstruction_penalty = False
    C_rec = theano.shared(floatX(1e3), name="C_rec")  # Hyperparameter of the reconstruction penalty

    # SVDD
    nu = theano.shared(floatX(.2), name="nu")
    c_mean_init = False
    c_mean_init_n_batches = "all"
    hard_margin = False
    block_coordinate = False
    k_update_epochs = 5  # update R and c only every k epochs, i.e. always train the network for k epochs in one block.
    R_update_solver = "minimize_scalar"  # "minimize_scalar" (default) or "lp" (linear program)
    R_update_scalar_method = "bounded"  # optimization method used in minimize_scalar ('brent', 'bounded', or 'golden')
    R_update_lp_obj = "primal" # on which objective ("primal" or "dual") should R be optimized if LP?
    center_fixed = True  # determine if center c should be fixed or not (in which case c is an optimization parameter)
    QP_solver = 'cvxopt'  # the library to use for solving the QP (or LP). One of ("cvxopt" or "gurobi")
    warm_up_n_epochs = 0  # iterations until R and c are also getting optimized

    # Data preprocessing
    if dataset in ("bdd100k", "prosivic", "dreyeve"):
        n_test_out  = n_test-n_test_in
        out_frac = floatX((n_test - n_test_out)/n_test)
    else:
        out_frac = floatX(.1)
    ad_experiment = True
    pca = False
    unit_norm_used = "l2"  # "l2" or "l1"
    gcn = False
    zca_whitening = False

    # MNIST parameters
    mnist_val_frac = 1./6
    mnist_bias = True
    mnist_rep_dim = 32
    mnist_architecture = 1  # choose one of the implemented architectures
    mnist_normal = 0
    mnist_outlier = -1

    # CIFAR-10 parameters
    cifar10_val_frac = 1./5
    cifar10_bias = True
    cifar10_rep_dim = 128
    cifar10_architecture = 1  # choose one of the implemented architectures
    cifar10_normal = 1
    cifar10_outlier = -1

    # GTSRB parameters
    gtsrb_rep_dim = 32

    # Plot parameters
    xp_path = "../log/" + dataset + "/"
    title_suffix = ""

    # SVM parameters
    svm_C = floatX(1.0)
    svm_nu = floatX(0.2)
    svm_GridSearchCV = False

    # KDE parameters
    kde_GridSearchCV = False

    # Diagnostics (should diagnostics be retrieved? Training is faster without)
    nnet_diagnostics = True  # diagnostics for neural networks in general (including Deep SVDD)
    e1_diagnostics = True  # diagnostics for neural networks in first epoch
    ae_diagnostics = True  # diagnostics for autoencoders
    print_options = False
