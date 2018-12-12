import argparse
import os
import sys
import theano
import time

from neuralnet import NeuralNet
from config import Configuration as Cfg
from utils.log import log_exp_config, log_NeuralNet, log_AD_results
from utils.visualization.diagnostics_plot import plot_diagnostics, plot_ae_diagnostics
from utils.visualization.filters_plot import plot_filters
from utils.visualization.images_plot import plot_outliers_and_most_normal
from utils.assertions import files_equal
from shutil import copyfile
from utils.monitoring import performance, ae_performance
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
# ====================================================================
# Parse arguments
# --------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    help="dataset name",
                    type=str, choices=["mnist", "cifar10", "gtsrb", "bdd100k", "dreyeve", "prosivic"])
parser.add_argument("--solver",
                    help="solver", type=str,
                    choices=["sgd", "momentum", "nesterov", "adagrad", "rmsprop", "adadelta", "adam", "adamax"])
parser.add_argument("--loss",
                    help="loss function",
                    type=str, choices=["ce", "svdd", "autoencoder"])
parser.add_argument("--lr",
                    help="initial learning rate",
                    type=float)
parser.add_argument("--lr_decay",
                    help="specify if learning rate should be decayed",
                    type=int, default=Cfg.lr_decay)
parser.add_argument("--lr_decay_after_epoch",
                    help="specify the epoch after learning rate should decay",
                    type=int, default=Cfg.lr_decay_after_epoch)
parser.add_argument("--lr_drop",
                    help="specify if learning rate should drop in a specified epoch",
                    type=int, default=Cfg.lr_drop)
parser.add_argument("--lr_drop_in_epoch",
                    help="specify the epoch in which learning rate should drop",
                    type=int, default=Cfg.lr_drop_in_epoch)
parser.add_argument("--lr_drop_factor",
                    help="specify the factor by which the learning rate should drop",
                    type=int, default=Cfg.lr_drop_factor)
parser.add_argument("--momentum",
                    help="momentum rate if optimization with momentum",
                    type=float, default=Cfg.momentum)
parser.add_argument("--block_coordinate",
                    help="specify if radius R and center c (if c is not fixed) should be solved for via the dual",
                    type=int, default=Cfg.block_coordinate)
parser.add_argument("--k_update_epochs",
                    help="update R and c in block coordinate descent only every k iterations",
                    type=int, default=Cfg.k_update_epochs)
parser.add_argument("--center_fixed",
                    help="specify if center c should be fixed or not",
                    type=int, default=Cfg.center_fixed)
parser.add_argument("--R_update_solver",
                    help="Solver for solving R",
                    type=str,
                    choices=["minimize_scalar", "lp"],
                    default=Cfg.R_update_solver)
parser.add_argument("--R_update_scalar_method",
                    help="Optimization method if minimize_scalar for solving R",
                    type=str,
                    choices=["brent", "bounded", "golden"],
                    default=Cfg.R_update_scalar_method)
parser.add_argument("--R_update_lp_obj",
                    help="Objective used for searching R in a block coordinate descent via LP (primal or dual)",
                    type=str,
                    choices=["primal", "dual"],
                    default=Cfg.R_update_scalar_method)
parser.add_argument("--warm_up_n_epochs",
                    help="specify the first epoch the QP solver should be applied",
                    type=int, default=Cfg.warm_up_n_epochs)
parser.add_argument("--use_batch_norm",
                    help="specify if Batch Normalization should be applied in the network",
                    type=int, default=Cfg.use_batch_norm)
parser.add_argument("--pretrain",
                    help="specify if weights should be pre-trained via autoenc",
                    type=int, default=Cfg.pretrain)
parser.add_argument("--nnet_diagnostics",
                    help="specify if diagnostics should be captured (faster training without)",
                    type=int, default=Cfg.nnet_diagnostics)
parser.add_argument("--e1_diagnostics",
                    help="specify if diagnostics of first epoch per batch should be captured",
                    type=int, default=Cfg.e1_diagnostics)
parser.add_argument("--ae_diagnostics",
                    help="specify if diagnostics should be captured in autoencoder (faster training without)",
                    type=int, default=Cfg.ae_diagnostics)
parser.add_argument("--ae_loss",
                    help="specify the reconstruction loss of the autoencoder",
                    type=str, default=Cfg.ae_loss)
parser.add_argument("--ae_lr_drop",
                    help="specify if learning rate should drop in a specified epoch",
                    type=int, default=Cfg.ae_lr_drop)
parser.add_argument("--ae_lr_drop_in_epoch",
                    help="specify the epoch in which learning rate should drop",
                    type=int, default=Cfg.ae_lr_drop_in_epoch)
parser.add_argument("--ae_lr_drop_factor",
                    help="specify the factor by which the learning rate should drop",
                    type=int, default=Cfg.ae_lr_drop_factor)
parser.add_argument("--ae_weight_decay",
                    help="specify if weight decay should be used in pretrain",
                    type=int, default=Cfg.ae_weight_decay)
parser.add_argument("--ae_C",
                    help="regularization hyper-parameter in pretrain",
                    type=float, default=Cfg.ae_C)
parser.add_argument("--batch_size",
                    help="batch size",
                    type=int, default=Cfg.batch_size)
parser.add_argument("--n_epochs",
                    help="number of epochs",
                    type=int)
parser.add_argument("--save_at",
                    help="number of epochs before saving model",
                    type=int, default=0)
parser.add_argument("--device",
                    help="Computation device to use for experiment",
                    type=str, default="cpu")
parser.add_argument("--xp_dir",
                    help="directory for the experiment",
                    type=str)
parser.add_argument("--in_name",
                    help="name for inputs of experiment",
                    type=str, default="")
parser.add_argument("--out_name",
                    help="name for outputs of experiment",
                    type=str, default="")
parser.add_argument("--leaky_relu",
                    help="specify if ReLU layer should be leaky",
                    type=int, default=Cfg.leaky_relu)
parser.add_argument("--weight_decay",
                    help="specify if weight decay should be used",
                    type=int, default=Cfg.weight_decay)
parser.add_argument("--C",
                    help="regularization hyper-parameter",
                    type=float, default=Cfg.C)
parser.add_argument("--reconstruction_penalty",
                    help="specify if a reconstruction (autoencoder) penalty should be used",
                    type=int, default=Cfg.reconstruction_penalty)
parser.add_argument("--C_rec",
                    help="reconstruction (autoencoder) penalty hyperparameter",
                    type=float, default=Cfg.C_rec)
parser.add_argument("--dropout",
                    help="specify if dropout layers should be applied",
                    type=int, default=Cfg.dropout)
parser.add_argument("--dropout_arch",
                    help="specify if dropout architecture should be used",
                    type=int, default=Cfg.dropout_architecture)
parser.add_argument("--c_mean_init",
                    help="specify if center c should be initialized as mean",
                    type=int, default=Cfg.c_mean_init)
parser.add_argument("--c_mean_init_n_batches",
                    help="from how many batches should the mean be computed?",
                    type=int, default=Cfg.c_mean_init_n_batches)  # default=-1 means "all"
parser.add_argument("--hard_margin",
                    help="Train deep SVDD with hard-margin algorithm",
                    type=int, default=0)
parser.add_argument("--nu",
                    help="nu parameter in one-class SVM",
                    type=float, default=0.1)
parser.add_argument("--out_frac",
                    help="fraction of outliers in data set",
                    type=float, default=0)
parser.add_argument("--seed",
                    help="numpy seed",
                    type=int, default=0)
parser.add_argument("--ad_experiment",
                    help="specify if experiment should be two- or multiclass",
                    type=int, default=1)
parser.add_argument("--weight_dict_init",
                    help="initialize first layer filters by dictionary",
                    type=int, default=0)
parser.add_argument("--pca",
                    help="apply pca in preprocessing",
                    type=int, default=0)
parser.add_argument("--unit_norm_used",
                    help="norm to use for scaling the data to unit norm",
                    type=str, default="l2")
parser.add_argument("--gcn",
                    help="apply global contrast normalization in preprocessing",
                    type=int, default=0)
parser.add_argument("--zca_whitening",
                    help="specify if data should be whitened",
                    type=int, default=0)
parser.add_argument("--mnist_val_frac",
                    help="specify the fraction the validation set of the initial training data should be",
                    type=float, default=1./6)
parser.add_argument("--mnist_bias",
                    help="specify if bias terms are used in MNIST network",
                    type=int, default=1)
parser.add_argument("--mnist_rep_dim",
                    help="specify the dimensionality of the last layer",
                    type=int, default=16)
parser.add_argument("--mnist_architecture",
                    help="specify which network architecture should be used",
                    type=int, default=1)
parser.add_argument("--mnist_normal",
                    help="specify normal class in MNIST",
                    type=int, default=0)
parser.add_argument("--mnist_outlier",
                    help="specify outlier class in MNIST",
                    type=int, default=1)
parser.add_argument("--cifar10_bias",
                    help="specify if bias terms are used in CIFAR-10 network",
                    type=int, default=1)
parser.add_argument("--cifar10_rep_dim",
                    help="specify the dimensionality of the last layer",
                    type=int, default=32)
parser.add_argument("--cifar10_architecture",
                    help="specify which network architecture should be used",
                    type=int, default=1)
parser.add_argument("--cifar10_normal",
                    help="specify normal class in CIFAR-10",
                    type=int, default=0)
parser.add_argument("--cifar10_outlier",
                    help="specify outlier class in CIFAR-10",
                    type=int, default=1)
parser.add_argument("--gtsrb_rep_dim",
                    help="specify the dimensionality of the last layer",
                    type=int, default=32)
parser.add_argument("--bdd100k_bias",
                    help="specify if bias terms are used in bdd100k network",
                    type=int, default=1)
parser.add_argument("--bdd100k_val_frac",
                    help="specify the fraction the validation set of the initial training data should be",
                    type=float, default=1./6)
parser.add_argument("--dreyeve_bias",
                    help="specify if bias terms are used in dreyeve network",
                    type=int, default=1)
parser.add_argument("--dreyeve_val_frac",
                    help="specify the fraction the validation set of the initial training data should be",
                    type=float, default=1./6)

'''
parser.add_argument("--bdd100k_rep_dim",
                    help="specify the dimensionality of the last layer",
                    type=int, default=32)
parser.add_argument("--bdd100k_architecture",
                    help="specify which network architecture should be used",
                    type=int, default=2)
parser.add_argument("--bdd100k_n_train",
                    help="number of images in training and validation sets",
                    type=int, default=1000)
parser.add_argument("--bdd100k_n_test",
                    help="number of images in training and validation sets",
                    type=int, default=1000)
'''
# ====================================================================


def main():

    args = parser.parse_args()
    if Cfg.print_options:
        print('Options:')
        for (key, value) in vars(args).iteritems():
            print("{:16}: {}".format(key, value))

    assert os.path.exists(args.xp_dir)

    # default value for basefile: string basis for all exported file names
    if args.out_name:
        base_file = "{}/{}".format(args.xp_dir, args.out_name)
    else:
        base_file = "{}/{}_{}_{}".format(args.xp_dir, args.dataset, args.solver, args.loss)

    # if pickle file already there, consider run already done
    if not Cfg.only_test and (os.path.exists("{}_weights.p".format(base_file)) and os.path.exists("{}_results.p".format(base_file))):
        sys.exit()

    # computation device
#    if 'gpu' in args.device:
#        theano.sandbox.cuda.use(args.device)

    # set save_at to n_epochs if not provided
    save_at = args.n_epochs if not args.save_at else args.save_at

    save_to = "{}_weights.p".format(base_file)
    weights = "{}/{}.p".format(args.xp_dir,args.in_name) if args.in_name else None
    print(weights)

    # update config data

    # plot parameters
    Cfg.xp_path = args.xp_dir

    # dataset
    Cfg.seed = args.seed
    Cfg.out_frac = args.out_frac
    Cfg.ad_experiment = bool(args.ad_experiment)
    Cfg.weight_dict_init = bool(args.weight_dict_init)
    Cfg.pca = bool(args.pca)
    Cfg.unit_norm_used = args.unit_norm_used
    Cfg.gcn = bool(args.gcn)
    Cfg.zca_whitening = bool(args.zca_whitening)
    Cfg.mnist_val_frac = args.mnist_val_frac
    Cfg.mnist_bias = bool(args.mnist_bias)
    Cfg.mnist_rep_dim = args.mnist_rep_dim
    Cfg.mnist_architecture = args.mnist_architecture
    Cfg.mnist_normal = args.mnist_normal
    Cfg.mnist_outlier = args.mnist_outlier
    Cfg.cifar10_bias = bool(args.cifar10_bias)
    Cfg.cifar10_rep_dim = args.cifar10_rep_dim
    Cfg.cifar10_architecture = args.cifar10_architecture
    Cfg.cifar10_normal = args.cifar10_normal
    Cfg.cifar10_outlier = args.cifar10_outlier
    Cfg.gtsrb_rep_dim = args.gtsrb_rep_dim
#    Cfg.bdd100k_rep_dim = args.bdd100k_rep_dim
#    Cfg.bdd100k_architecture = args.bdd100k_architecture
#    Cfg.bdd100k_val_frac = args.bdd100k_val_frac
#    Cfg.bdd100k_bias = args.bdd100k_bias
#    Cfg.bdd100k_n_train = args.bdd100k_n_train
#    Cfg.bdd100k_n_test = args.bdd100k_n_test

    # neural network
    Cfg.softmax_loss = (args.loss == 'ce')
    Cfg.svdd_loss = (args.loss == 'svdd')
    Cfg.reconstruction_loss = (args.loss == 'autoencoder')
    Cfg.use_batch_norm = bool(args.use_batch_norm)
    Cfg.learning_rate.set_value(args.lr)
    Cfg.lr_decay = bool(args.lr_decay)
    Cfg.lr_decay_after_epoch = args.lr_decay_after_epoch
    Cfg.lr_drop = bool(args.lr_drop)
    Cfg.lr_drop_in_epoch = args.lr_drop_in_epoch
    Cfg.lr_drop_factor = args.lr_drop_factor
    Cfg.momentum.set_value(args.momentum)
    if args.solver == "rmsprop":
        Cfg.rho.set_value(0.9)
    if args.solver == "adadelta":
        Cfg.rho.set_value(0.95)
    Cfg.block_coordinate = bool(args.block_coordinate)
    Cfg.k_update_epochs = args.k_update_epochs
    Cfg.center_fixed = bool(args.center_fixed)
    Cfg.R_update_solver = args.R_update_solver
    Cfg.R_update_scalar_method = args.R_update_scalar_method
    Cfg.R_update_lp_obj = args.R_update_lp_obj
    Cfg.warm_up_n_epochs = args.warm_up_n_epochs
    Cfg.batch_size = args.batch_size
    Cfg.leaky_relu = bool(args.leaky_relu)

    # Pre-training and autoencoder configuration
    Cfg.pretrain = bool(args.pretrain)
    Cfg.ae_loss = args.ae_loss
    Cfg.ae_lr_drop = bool(args.ae_lr_drop)
    Cfg.ae_lr_drop_in_epoch = args.ae_lr_drop_in_epoch
    Cfg.ae_lr_drop_factor = args.ae_lr_drop_factor
    Cfg.ae_weight_decay = bool(args.ae_weight_decay)
    Cfg.ae_C.set_value(args.ae_C)

    # SVDD parameters
    Cfg.nu.set_value(args.nu)
    Cfg.c_mean_init = bool(args.c_mean_init)
    if args.c_mean_init_n_batches == -1:
        Cfg.c_mean_init_n_batches = "all"
    else:
        Cfg.c_mean_init_n_batches = args.c_mean_init_n_batches
    Cfg.hard_margin = bool(args.hard_margin)

    # regularization
    Cfg.weight_decay = bool(args.weight_decay)
    Cfg.C.set_value(args.C)
    Cfg.reconstruction_penalty = bool(args.reconstruction_penalty)
    Cfg.C_rec.set_value(args.C_rec)
    Cfg.dropout = bool(args.dropout)
    Cfg.dropout_architecture = bool(args.dropout_arch)

    # diagnostics
    Cfg.nnet_diagnostics = bool(args.nnet_diagnostics)
    Cfg.e1_diagnostics = bool(args.e1_diagnostics)
    Cfg.ae_diagnostics = bool(args.ae_diagnostics)

    # Check for previous copy of configuration and compare, abort if not equal
    logged_config = args.xp_dir+"/configuration.py"
    current_config = "./config.py"
    if os.path.exists(logged_config):
        print("Comparing logged and current config")
#        assert(files_equal(logged_config,current_config, "dataset ="))
    else:
        copyfile(current_config,logged_config)

    if not Cfg.only_test: # Run original DSVDD code, both training and testing in one
        # train
        # load from checkpoint if available
        
        start_new_nnet = False
        if os.path.exists(args.xp_dir+"/ae_pretrained_weights.p"):
                print("Pretrained AE found")
                Cfg.pretrain = False
                nnet = NeuralNet(dataset=args.dataset, use_weights=args.xp_dir+"/ae_pretrained_weights.p", pretrain=False)
        elif Cfg.pretrain:
            if os.path.exists(args.xp_dir+"/ae_checkpoint.p"):
                print("AE checkpoint found, resuming training")
                nnet = NeuralNet(dataset=args.dataset, use_weights=args.xp_dir+"/ae_checkpoint.p", pretrain=True)
            else:
                start_new_nnet = True

        elif os.path.exists(args.xp_dir+"/checkpoint.p"):
            print("DSVDD checkpoint found, resuming training")
            nnet = NeuralNet(dataset=args.dataset, use_weights=args.xp_dir+"/checkpoint.p", pretrain=False)
        else:
            start_new_nnet = True

        if start_new_nnet:
            nnet = NeuralNet(dataset=args.dataset, use_weights=weights, pretrain=Cfg.pretrain)

        # pre-train weights via autoencoder, if specified
        if Cfg.pretrain:
            nnet.pretrain(solver="adam", lr=Cfg.pretrain_learning_rate, n_epochs=Cfg.n_pretrain_epochs)

        nnet.train(solver=args.solver, n_epochs=args.n_epochs, save_at=save_at, save_to=save_to)

        # pickle/serialize AD results
        if Cfg.ad_experiment:
            nnet.log_results(filename=Cfg.xp_path + "/AD_results.p")

        # text log
        nnet.log.save_to_file("{}_results.p".format(base_file))  # save log
        log_exp_config(Cfg.xp_path, args.dataset)
        log_NeuralNet(Cfg.xp_path, args.loss, args.solver, args.lr, args.momentum, None, args.n_epochs, args.C, args.C_rec,
                    args.nu, args.dataset)
        if Cfg.ad_experiment:
            log_AD_results(Cfg.xp_path, nnet)

        # plot diagnostics
        if Cfg.nnet_diagnostics:
            # common suffix for plot titles
            str_lr = "lr = " + str(args.lr)
            C = int(args.C)
            if not Cfg.weight_decay:
                C = None
            str_C = "C = " + str(C)
            Cfg.title_suffix = "(" + args.solver + ", " + str_C + ", " + str_lr + ")"

            if args.loss == 'autoencoder':
                plot_ae_diagnostics(nnet, Cfg.xp_path, Cfg.title_suffix)
            else:
                plot_diagnostics(nnet, Cfg.xp_path, Cfg.title_suffix)

        if Cfg.plot_filters:
            print("Plotting filters")
            plot_filters(nnet, Cfg.xp_path, Cfg.title_suffix)

        # If AD experiment, plot most anomalous and most normal
        if Cfg.ad_experiment and Cfg.plot_most_out_and_norm:
            n_img = 32
            plot_outliers_and_most_normal(nnet, n_img, Cfg.xp_path)

    else: # Load previous network and run only test 

        # Load parameters from previous training
        ae_net = NeuralNet(dataset=args.dataset, use_weights=args.xp_dir+"/ae_pretrained_weights.p", pretrain=True)
        ae_net.ae_solver = args.solver.lower()
        #ae_net.ae_learning_rate = args.lr
        #ae_net.ae_n_epochs = args.n_epochs

        # set learning rate
        #lr_tmp = Cfg.learning_rate.get_value()
        #Cfg.learning_rate.set_value(Cfg.floatX(lr))
        ae_net.compile_autoencoder()
        _, recon_errors = ae_performance(ae_net, 'test')
        print("Computed reconstruction errors")


        nnet = NeuralNet(dataset=args.dataset, use_weights="{}/weights_best_ep.p".format(args.xp_dir))
        nnet.solver = args.solver.lower()
        nnet.compile_updates()
        # nnet.evaluate(solver = args.solver)
        # nnet.test_time = time.time() - nnet.clock
        # # pickle/serialize AD results
        # if Cfg.ad_experiment:
        #     nnet.log_results(filename=Cfg.xp_path + "/AD_results.p")

        # TODO retrieve labels and scores from evaluation
        _, _, dsvdd_scores = performance(nnet,'test')

        labels = nnet.data._y_test

        # # text log
        # nnet.log.save_to_file("{}_results.p".format(base_file))  # save log
        # log_exp_config(Cfg.xp_path, args.dataset)
        # log_NeuralNet(Cfg.xp_path, args.loss, args.solver, args.lr, args.momentum, None, args.n_epochs, args.C, args.C_rec,
        #             args.nu, args.dataset)
        # if Cfg.ad_experiment:
        #     log_AD_results(Cfg.xp_path, nnet)

        # Save scores and labels for comparison with other experiments
        
        if Cfg.export_results:
            for name in ("", "_recon_err"):
                results_filepath = '/home/exjobb_resultat/data/%s_DSVDD%s.pkl'%(args.dataset,name)
                with open(results_filepath,'wb') as f:
                    if name is "_recon_err":
                        pickle.dump([recon_errors,labels],f)
                    else:
                        pickle.dump([dsvdd_scores,labels],f)
                print("Saved results to %s"%results_filepath)

                # Update data source dict with experiment name
                common_results_dict = pickle.load(open('/home/exjobb_resultat/data/name_dict.pkl','rb'))
                exp_name = args.xp_dir.strip('../log/%s/'%args.dataset)
                common_results_dict[args.dataset]["DSVDD%s"%name] = exp_name
                pickle.dump(common_results_dict,open('/home/exjobb_resultat/data/name_dict.pkl','wb'))

        # print test results to console
        print("\nOutliers from %s"%Cfg.test_out_folder)
        print("%d inliers, %d outliers"%(Cfg.n_test_in, Cfg.n_test-Cfg.n_test_in))
        print("Test results:\n")
        print("\t\tAUROC\tAUPRC\n")

        # Compute test metrics before printing
        auroc_recon = roc_auc_score(labels, recon_errors)
        auroc_dsvdd = roc_auc_score(labels, dsvdd_scores)
        pr, rc, _ = precision_recall_curve(labels, recon_errors)
        auprc_recon = auc(rc,pr)
        pr, rc, _ = precision_recall_curve(labels, dsvdd_scores)
        auprc_dsvdd = auc(rc,pr)

        print("Recon.err:\t%.4f\t%.4f"%(auroc_recon,auprc_recon))
        print("DSVDD:\t\t%.4f\t%.4f"%(auroc_dsvdd,auprc_dsvdd))

if __name__ == '__main__':
    main()
