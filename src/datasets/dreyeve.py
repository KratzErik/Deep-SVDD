from datasets.base import DataLoader
from datasets.preprocessing import center_data, normalize_data, rescale_to_unit_interval, \
    global_contrast_normalization, zca_whitening, extract_norm_and_out, learn_dictionary, pca
from utils.visualization.mosaic_plot import plot_mosaic
from utils.misc import flush_last_line
from config import Configuration as Cfg
from datasets.modules import addConvModule
import os
import numpy as np
import cPickle as pickle
from keras.preprocessing.image import load_img, img_to_array

class DREYEVE_DataLoader(DataLoader):

    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "dreyeve"

        self.n_train = Cfg.dreyeve_n_train
        self.n_val = Cfg.dreyeve_n_val
        self.n_test = Cfg.dreyeve_n_test

#        self.out_frac = Cfg.dreyeve_out_frac

        self.image_height = Cfg.dreyeve_image_height
        self.image_width = Cfg.dreyeve_image_width
        self.channels = Cfg.dreyeve_channels

        self.seed = Cfg.seed

        if Cfg.ad_experiment:
            self.n_classes = 2
        else:
            self.n_classes = 6 #there are 6 different weather types, however this should not be used

        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

        self.data_path = './data/dreyeve'
#        self.norm_filenames = Cfg.dreyeve_norm_filenames
#        self.out_filenames = Cfg.dreyeve_out_filenames

#        self.label_path = Cfg.dreyeve_labels_file
#        self.attributes_normal = Cfg.dreyeve_attributes_normal
#        self.attributes_outlier = Cfg.dreyeve_attributes_outlier

#        self.save_name_lists = Cfg.dreyeve_save_name_lists
#        self.get_norm_and_out_sets = Cfg.dreyeve_get_norm_and_out_sets


        self.on_memory = True
        Cfg.store_on_gpu = True

        # load data from disk
        self.load_data()
        if Cfg.dreyeve_architecture not in (1,2,3,4):
            self.print_architecture()

    def check_specific(self):

        # store primal variables on RAM
        assert Cfg.store_on_gpu

    def load_data(self, original_scale=False):

        print("Loading data...")

        # load normal and outlier data
        tmp = [load_img(Cfg.dreyeve_train_folder + filename) for filename in os.listdir(Cfg.dreyeve_train_folder)]
        print(type(tmp[0]))
        self._X_train = [img_to_array(load_img(Cfg.dreyeve_train_folder + filename)) for filename in os.listdir(Cfg.dreyeve_train_folder)]
        self._X_val = [img_to_array(load_img(Cfg.dreyeve_val_folder + filename)) for filename in os.listdir(Cfg.dreyeve_val_folder)]
        self._X_test = [img_to_array(load_img(Cfg.dreyeve_test_folder + filename)) for filename in os.listdir(Cfg.dreyeve_test_folder)]
        self._y_test = np.concatenate([np.zeros((Cfg.dreyeve_n_test_in,),dtype=np.int32),np.ones((Cfg.dreyeve_n_test-Cfg.dreyeve_n_test_in,),dtype=np.int32)])
        
        if Cfg.dreyeve_test_in_loc != "first": # outliers before inliers in test set
            self._y_test = self._y_test[::-1] #  flip labels

        # tranpose to channels first
        self._X_train = np.moveaxis(self._X_train,-1,1)
        self._X_val = np.moveaxis(self._X_val,-1,1)
        self._X_test = np.moveaxis(self._X_test,-1,1)


        # cast data properly
        self._X_train = self._X_train.astype(np.float32)
        self._X_val = self._X_val.astype(np.float32)
        self._X_test = self._X_test.astype(np.float32)
        self._y_test = self._y_test.astype(np.int32)

        # Train and val labels are 0, since all are normal class
        self._y_train = np.zeros((len(self._X_train),),dtype=np.int32)
        self._y_val = np.zeros((len(self._X_val),),dtype=np.int32)

        if Cfg.ad_experiment:
            # shuffle to obtain random validation splits
            np.random.seed(self.seed)

            # shuffle data (since batches are extracted block-wise)
            self.n_train = len(self._y_train)
            self.n_val = len(self._y_val)
            perm_train = np.random.permutation(self.n_train)
            perm_val = np.random.permutation(self.n_val)
            self._X_train = self._X_train[perm_train]
            self._y_train = self._y_train[perm_train]
            self._X_val = self._X_train[perm_val]
            self._y_val = self._y_train[perm_val]

            # Subset train set such that we only get batches of the same size
            assert(self.n_train >= Cfg.batch_size)
            self.n_train = (self.n_train / Cfg.batch_size) * Cfg.batch_size
            subset = np.random.choice(len(self._X_train), self.n_train, replace=False)
            self._X_train = self._X_train[subset]
            self._y_train = self._y_train[subset]

            # Adjust number of batches
            Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

        # normalize data (if original scale should not be preserved)
        if not original_scale:

            # simple rescaling to [0,1]
            normalize_data(self._X_train, self._X_val, self._X_test, scale=np.float32(255))

            # global contrast normalization
            if Cfg.gcn:
                global_contrast_normalization(self._X_train, self._X_val, self._X_test, scale=Cfg.unit_norm_used)

            # ZCA whitening
            if Cfg.zca_whitening:
                self._X_train, self._X_val, self._X_test = zca_whitening(self._X_train, self._X_val, self._X_test)

            # rescale to [0,1] (w.r.t. min and max in train data)
            rescale_to_unit_interval(self._X_train, self._X_val, self._X_test)

            # PCA
            if Cfg.pca:
                self._X_train, self._X_val, self._X_test = pca(self._X_train, self._X_val, self._X_test, 0.95)

        flush_last_line()
        print("Data loaded.")

    def print_architecture(self):
        tmp = Cfg.dreyeve_architecture.split("_")
        use_pool = int(tmp[0]) # 1 or 0
        n_conv = int(tmp[1])
        n_dense = int(tmp[2])
        c1 = int(tmp[3])
        zsize = int(tmp[4])
        ksize= int(tmp[5])
        stride = int(tmp[6])
        pad = int(tmp[7])

        print("Architecture:\n")
        print("Conv. layers: %d"%n_conv)
        print("Dense layers: %d"%n_dense)
        print("Channels out of first conv_layer: %d"%c1)
        print("Latent dim: %d"%zsize)
        print("Kernel, stride, pad: %d, %d, %d" % (ksize, stride, pad))



    def build_architecture(self, nnet):
        # implementation of different network architectures
        if Cfg.dreyeve_architecture not in (1,2,3):
            # architecture spec A_B_C_D_E_F_G_H
            tmp = Cfg.dreyeve_architecture.split("_")
            use_pool = int(tmp[0]) # 1 or 0
            n_conv = int(tmp[1])
            n_dense = int(tmp[2])
            c1 = int(tmp[3])
            zsize = int(tmp[4])
            ksize= int(tmp[5])
            stride = int(tmp[6])
            pad = int(tmp[7])
            num_filters = c1

            # If using maxpool, we should have pad = same
            if use_pool:
                pad = 'same'
            else:
                pad = (pad,pad)

            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, n_filters=8, filter_size=5, n_sample=Cfg.dreyeve_n_dict_learn)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))
            else:
                W1_init = None

            # Build architecture
            nnet.addInputLayer(shape=(None, self.channels, self.image_height, self.image_width))

            # Add all but last conv. layer
            for i in range(n_conv-1):
                addConvModule(nnet,
                          num_filters=num_filters,
                          filter_size=(ksize,ksize),
                          W_init=W1_init,
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          dropout=Cfg.dropout,
                          p_dropout=0.2,
                          use_maxpool = use_pool,
                          stride = stride,
                          pad = pad,
                          )
                num_filters *= 2

                print("Added conv_layer %d" % nnet.n_conv_layers)

            if n_dense > 0:
                addConvModule(nnet,
                          num_filters=num_filters,
                          filter_size=(ksize,ksize),
                          W_init=W1_init,
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          dropout=Cfg.dropout,
                          p_dropout=0.2,
                          use_maxpool = use_pool,
                          stride = stride,
                          pad = pad,
                          )
                print("Added conv_layer %d" % nnet.n_conv_layers)

                #shape_in = (None,self.image_height,W_in,num_filters)
                # Dense layer
                if Cfg.dropout:
                    nnet.addDropoutLayer()
                if Cfg.dreyeve_bias:
                    nnet.addDenseLayer(num_units=zsize)
                else:
                    nnet.addDenseLayer(num_units=zsize,
                                        b=None)
                print("Added dense layer")

            else:
                h = self.image_height / (2**(n_conv-1))
                addConvModule(nnet,
                          num_filters=zsize,
                          filter_size=(h,h),
                          W_init=W1_init,
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=False,
                          dropout=False,
                          p_dropout=0.2,
                          use_maxpool = False,
                          stride = (1,1),
                          pad = (0,0),
                          )

                print("Added conv_layer %d" % nnet.n_conv_layers)

        elif Cfg.dreyeve_architecture == 1: # For 256by256 images

            if Cfg.dropout:
                units_multiplier = 2
            else:
                units_multiplier = 1

            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, n_filters=8, filter_size=5, n_sample=Cfg.dreyeve_n_dict_learn)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))
            else:
                W1_init = None

            # build architecture
            nnet.addInputLayer(shape=(None, self.channels, self.image_height, self.image_width))

            # conv1 : h_in 256 -> h_out 128
            addConvModule(nnet,
                          num_filters=16 * units_multiplier,
                          filter_size=(5,5),
                          W_init=W1_init,
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          dropout=Cfg.dropout,
                          p_dropout=0.2)
            
            # conv2 : h_in 128 -> h_out 64
            addConvModule(nnet,
                          num_filters=32 * units_multiplier,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          dropout=Cfg.dropout)
            
            # conv3 : h_in 64 -> h_out 32
            addConvModule(nnet,
                          num_filters=64 * units_multiplier,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          dropout=Cfg.dropout)

            # conv4 : h_in 32 -> h_out 16
            addConvModule(nnet,
                          num_filters=64 * units_multiplier,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          dropout=Cfg.dropout)

            # conv5 : h_in 16 -> h_out 8
            addConvModule(nnet,
                          num_filters=128 * units_multiplier,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          dropout=Cfg.dropout)

            # conv6 : h_in 8 -> h_out 4
            addConvModule(nnet,
                          num_filters=256 * units_multiplier,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          dropout=Cfg.dropout)

            # Dense layer
            if Cfg.dropout:
                nnet.addDropoutLayer()
            if Cfg.dreyeve_bias:
                nnet.addDenseLayer(num_units=Cfg.dreyeve_rep_dim * units_multiplier)
            else:
                nnet.addDenseLayer(num_units=Cfg.dreyeve_rep_dim * units_multiplier,
                                   b=None)

        elif Cfg.dreyeve_architecture == 3:
            
            #(192,320) input: (2,2) maxpooling down to (3,5)-image before dense layer

            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, n_filters=16, filter_size=5, n_sample=Cfg.dreyeve_n_dict_learn)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))
            else:
                W1_init = None

            # build architecture 1

            # input layer
            nnet.addInputLayer(shape=(None, self.channels, self.image_height, self.image_width))

            # convlayer 1
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same')
            else:
                if Cfg.weight_dict_init & (not nnet.pretrained):
                    nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                      W=W1_init, b=None)
                else:
                    nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                      b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #pool 1
            nnet.addMaxPool(pool_size=(2, 2))

            # convlayer 2
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            #pool 2
            nnet.addMaxPool(pool_size=(2, 2))

            # convlayer 3
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #pool 3
            nnet.addMaxPool(pool_size=(2, 2))

            # convlayer 4
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #pool4
            nnet.addMaxPool(pool_size=(2, 2))

            # convlayer 5
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            
            # pool 5
            nnet.addMaxPool(pool_size=(2, 2))

            # convlayer 6
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=256, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=256, filter_size=(5, 5),  pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            # pool 6
            nnet.addMaxPool(pool_size=(2, 2))

            # dense layer 1
            if Cfg.dreyeve_bias:
                nnet.addDenseLayer(num_units=Cfg.dreyeve_rep_dim)
            else:
                nnet.addDenseLayer(num_units=Cfg.dreyeve_rep_dim, b=None)


        elif Cfg.dreyeve_architecture == 4:
            # (192,320) input: first pooling is (3,5), then (2,2) pooling down to (4,4)-image just as for CIFAR-10

            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, n_filters=16, filter_size=5, n_sample=Cfg.dreyeve_n_dict_learn)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))
            else:
                W1_init = None

            # build architecture 1

            # input layer
            nnet.addInputLayer(shape=(None, self.channels, self.image_height, self.image_width))

            # convlayer 1
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same')
            else:
                if Cfg.weight_dict_init & (not nnet.pretrained):
                    nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                      W=W1_init, b=None)
                else:
                    nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                      b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #pool 1
            nnet.addMaxPool(pool_size=(3, 5))

            # convlayer 2
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            #pool 2
            nnet.addMaxPool(pool_size=(2, 2))

            # convlayer 3
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #pool 3
            nnet.addMaxPool(pool_size=(2, 2))

            # convlayer 4
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #pool4
            nnet.addMaxPool(pool_size=(2, 2))

            # convlayer 5
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            
            # pool 5
            nnet.addMaxPool(pool_size=(2, 2))

            # convlayer 6
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=256, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=256, filter_size=(5, 5),  pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            # pool 6
            nnet.addMaxPool(pool_size=(2, 2))

            # dense layer 1
            if Cfg.dreyeve_bias:
                nnet.addDenseLayer(num_units=Cfg.dreyeve_rep_dim)
            else:
                nnet.addDenseLayer(num_units=Cfg.dreyeve_rep_dim, b=None)

        else:
            raise ValueError("No valid choice of architecture")

        # Add ouput/feature layer
        if Cfg.softmax_loss:
            nnet.addDenseLayer(num_units=1)
            nnet.addSigmoidLayer()
        elif Cfg.svdd_loss:
            nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
        else:
            raise ValueError("No valid choice of loss for dataset " + self.dataset_name)

    def build_autoencoder(self, nnet):

        # implementation of different network architectures
        if Cfg.dreyeve_architecture not in (1,2,3):
            # architecture spec A_B_C_D_E_F_G_H
            tmp = Cfg.dreyeve_architecture.split("_")
            use_pool = int(tmp[0]) # 1 or 0
            n_conv = int(tmp[1])
            n_dense = int(tmp[2])
            c_out = int(tmp[3])
            zsize = int(tmp[4])
            ksize= int(tmp[5])
            stride = int(tmp[6])
            pad = int(tmp[7])
            num_filters = c_out

            if use_pool:
                pad = 'same'
            else:
                pad = (pad,pad)

            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, n_filters=8, filter_size=5, n_sample=Cfg.dreyeve_n_dict_learn)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))
            else:
                W1_init = None

            # Build architecture
            nnet.addInputLayer(shape=(None, self.channels, self.image_height, self.image_width))

            # Add all but last conv. layer
            for i in range(n_conv-1):
                addConvModule(nnet,
                          num_filters=num_filters,
                          filter_size=(ksize,ksize),
                          W_init=W1_init,
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          dropout=Cfg.dropout,
                          p_dropout=0.2,
                          use_maxpool = use_pool,
                          stride = stride,
                          pad = pad,
                          )

                num_filters *= 2

                print("Added conv_layer %d" % nnet.n_conv_layers)

            if n_dense > 0:
                addConvModule(nnet,
                          num_filters=num_filters,
                          filter_size=(ksize,ksize),
                          W_init=W1_init,
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          dropout=Cfg.dropout,
                          p_dropout=0.2,
                          use_maxpool = use_pool,
                          stride = stride,
                          pad = pad,
                          )
                print("Added conv_layer %d" % nnet.n_conv_layers)
                # Dense layer
                if Cfg.dropout:
                    nnet.addDropoutLayer()
                if Cfg.dreyeve_bias:
                    nnet.addDenseLayer(num_units=zsize)
                else:
                    nnet.addDenseLayer(num_units=zsize,
                                        b=None)
                print("Added dense layer")
            else:
                h = self.image_height / (2**(n_conv-1))
                addConvModule(nnet,
                          num_filters=zsize,
                          filter_size=(h,h),
                          W_init=W1_init,
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=False,
                          dropout=False,
                          p_dropout=0.2,
                          use_maxpool = False,
                          stride = (1,1),
                          pad = (0,0),
                          )

                print("Added conv_layer %d" % nnet.n_conv_layers)

            nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
            print("Feature layer here")

            n_deconv_layers = 0
            if n_dense > 0:

                h1 = self.image_height // (2**n_conv) # height = width of image going into first conv layer
                num_filters =  c_out * (2**(n_conv-1))
                nnet.addDenseLayer(num_units = h1**2 * num_filters)
                nnet.addReshapeLayer(shape=([0], num_filters, h1, h1))
                num_filters = num_filters // 2

                nnet.addUpscale(scale_factor=(2,2)) # since maxpool is after each conv. each upscale is before corresponding deconv

                addConvModule(nnet,
                          num_filters=num_filters,
                          filter_size=(ksize,ksize),
                          W_init=W1_init,
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          dropout=Cfg.dropout,
                          p_dropout=0.2,
                          use_maxpool = False,
                          stride = stride,
                          pad = pad,
                          upscale = True
                          )
                n_deconv_layers += 1
                print("Added deconv_layer %d" % n_deconv_layers)

                num_filters //=2
            else:
                nnet.addUpscale(scale_factor=(2,2)) # since maxpool is after each conv. each upscale is before corresponding deconv

                h2 = self.image_height // (2**(n_conv-1)) # height of image going in to second conv layer
                num_filters = c_out * (2**(n_conv-2))
                addConvModule(nnet,
                          num_filters=num_filters,
                          filter_size=(h2,h2),
                          W_init=W1_init,
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          dropout=Cfg.dropout,
                          p_dropout=0.2,
                          use_maxpool = False,
                          stride = (1,1),
                          pad = (0,0),
                          upscale = True
                          )
                n_deconv_layers += 1
                print("Added deconv_layer %d" % n_deconv_layers)

            # Add remaining deconv layers
            for i in range(n_conv-2):
                addConvModule(nnet,
                          num_filters=num_filters,
                          filter_size=(ksize,ksize),
                          W_init=W1_init,
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          dropout=Cfg.dropout,
                          p_dropout=0.2,
                          use_maxpool = False,
                          stride = stride,
                          pad = pad,
                          upscale = True
                          )
                n_deconv_layers += 1
                print("Added deconv_layer %d" % n_deconv_layers)
                num_filters //=2

            # add reconstruction layer
            # reconstruction
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(num_filters=self.channels,
                                  filter_size=(ksize, ksize),
                                  pad='same')
            else:
                nnet.addConvLayer(num_filters=self.channels,
                                  filter_size=(ksize, ksize),
                                  pad='same',
                                  b=None)
            print("Added reconstruction layer")

        if Cfg.dreyeve_architecture == 1:
            first_layer_n_filters = 16
            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, first_layer_n_filters, 5, n_sample=Cfg.dreyeve_n_dict_learn)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))
            else:
                W1_init = None

            nnet.addInputLayer(shape=(None, self.channels, self.image_height, self.image_width))

            addConvModule(nnet,
                          num_filters=first_layer_n_filters,
                          filter_size=(5,5),
                          W_init=W1_init,
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm)

            addConvModule(nnet,
                          num_filters=32,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm)

            addConvModule(nnet,
                          num_filters=64,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm)

            addConvModule(nnet,
                          num_filters=64,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm)

            addConvModule(nnet,
                          num_filters=128,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm)

            addConvModule(nnet,
                          num_filters=256,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm)

            # Code Layer
            if Cfg.dreyeve_bias:
                nnet.addDenseLayer(num_units=Cfg.dreyeve_rep_dim)
            else:
                nnet.addDenseLayer(num_units=Cfg.dreyeve_rep_dim, b=None)
            nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
            nnet.addReshapeLayer(shape=([0], (Cfg.dreyeve_rep_dim / 16), 4, 4))
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            #nnet.addUpscale(scale_factor=(2,2))  # TODO: is this Upscale necessary? Shouldn't there be as many Upscales as MaxPools?

            # Deconv and unpool 1
            addConvModule(nnet,
                          num_filters=128,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          upscale=True)

            # Deconv and unpool 2
            addConvModule(nnet,
                          num_filters=64,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          upscale=True)

            # Deconv and unpool 3
            addConvModule(nnet,
                          num_filters=64,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          upscale=True)

            # Deconv and unpool 4
            addConvModule(nnet,
                          num_filters=64,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          upscale=True)

            # Deconv and unpool 5
            addConvModule(nnet,
                          num_filters=64,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          upscale=True)

            # Deconv and unpool 6
            addConvModule(nnet,
                          num_filters=64,
                          filter_size=(5,5),
                          bias=Cfg.dreyeve_bias,
                          pool_size=(2,2),
                          use_batch_norm=Cfg.use_batch_norm,
                          upscale=True)

            # reconstruction
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(num_filters=self.channels,
                                  filter_size=(5, 5),
                                  pad='same')
            else:
                nnet.addConvLayer(num_filters=self.channels,
                                  filter_size=(5, 5),
                                  pad='same',
                                  b=None)
            nnet.addSigmoidLayer()


        if Cfg.dreyeve_architecture == 3:
        #(192,320) input: (2,2) maxpooling down to (3,5)-image before dense layer

            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, n_filters=16, filter_size=5, n_sample=Cfg.dreyeve_n_dict_learn)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))
            else:
                W1_init = None

            # build architecture 1

            # input layer
            nnet.addInputLayer(shape=(None, self.channels, self.image_height, self.image_width))

            # convlayer 1
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same')
            else:
                if Cfg.weight_dict_init & (not nnet.pretrained):
                    nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                      W=W1_init, b=None)
                else:
                    nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                      b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #pool 1
            nnet.addMaxPool(pool_size=(2, 2))

            # convlayer 2
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            #pool 2
            nnet.addMaxPool(pool_size=(2, 2))

            # convlayer 3
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #pool 3
            nnet.addMaxPool(pool_size=(2, 2))

            # convlayer 4
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #pool4
            nnet.addMaxPool(pool_size=(2, 2))

            # convlayer 5
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            
            # pool 5
            nnet.addMaxPool(pool_size=(2, 2))

            # convlayer 6
            if Cfg.dreyeve_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=256, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=256, filter_size=(5, 5),  pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            # pool 6
            nnet.addMaxPool(pool_size=(2, 2))

            # shape is now (3,5) images with 256 channels
            # Code Layer
            nnet.addDenseLayer(num_units=Cfg.dreyeve_rep_dim, b=None)
            nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
            nnet.addReshapeLayer(shape=([0], (Cfg.dreyeve_rep_dim / (3*5)), 3, 5))
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            # unpool1
            nnet.addUpscale(scale_factor=(2, 2))

            # deconv 1
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            # unpool2
            nnet.addUpscale(scale_factor=(2, 2))

            # deconv2
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            # unpool3    
            nnet.addUpscale(scale_factor=(2, 2))

            #deconv3
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #unpool4
            nnet.addUpscale(scale_factor=(2, 2))

            #deconv4
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #unpool5
            nnet.addUpscale(scale_factor=(2, 2))

            #deconv5
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #unpool6
            nnet.addUpscale(scale_factor=(2, 2))
            

            #deconv6
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=self.channels, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            nnet.addSigmoidLayer()




        if Cfg.dreyeve_architecture == 4:

            # input (256,256)
            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, 16, 5, n_sample=Cfg.dreyeve_n_dict_learn)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))

            #input
            nnet.addInputLayer(shape=(None, 3, self.image_height,self.image_width))

            #conv1
            if Cfg.weight_dict_init & (not nnet.pretrained):
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                  W=W1_init, b=None)
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                  b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #pool1
            nnet.addMaxPool(pool_size=(2, 2))

            #conv2
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            
            #pool2
            nnet.addMaxPool(pool_size=(2, 2))

            #conv3
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            
            #pool3
            nnet.addMaxPool(pool_size=(2, 2))

            #conv4
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            
            #pool4
            nnet.addMaxPool(pool_size=(2, 2))

            #conv5
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            
            #pool5
            nnet.addMaxPool(pool_size=(2, 2))

            #conv6
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            
            #pool6
            nnet.addMaxPool(pool_size=(2, 2))

            #conv7
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=256, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            
            #pool7
            nnet.addMaxPool(pool_size=(2, 2))


            # shape is now (2,2) images
            # Code Layer
            nnet.addDenseLayer(num_units=Cfg.dreyeve_rep_dim, b=None)
            nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
            nnet.addReshapeLayer(shape=([0], (Cfg.dreyeve_rep_dim / (2*2)), 2, 2))
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            # unpool1
            nnet.addUpscale(scale_factor=(2, 2))

            # deconv 1
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=256, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            # unpool2
            nnet.addUpscale(scale_factor=(2, 2))

            # deconv2
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            # unpool3    
            nnet.addUpscale(scale_factor=(2, 2))

            #deconv3
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #unpool4
            nnet.addUpscale(scale_factor=(2, 2))

            #deconv4
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #unpool5
            nnet.addUpscale(scale_factor=(2, 2))

            #deconv5
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #unpool6
            nnet.addUpscale(scale_factor=(2, 2))

            #deconv6
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #unpool7
            nnet.addUpscale(scale_factor=(2, 2))

            #deconv7
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #unpool7
            nnet.addUpscale(scale_factor=(2, 2))

            #deconv8
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=self.channels, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            nnet.addSigmoidLayer()

        if Cfg.dreyeve_architecture == 2:

            # input (256,256)
            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionry(nnet.data._X_train, 16, 5, n_sample=500)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))

            #input
            nnet.addInputLayer(shape=(None, 3, self.image_height,self.image_width))

            #conv1
            if Cfg.weight_dict_init & (not nnet.pretrained):
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                  W=W1_init, b=None)
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                                  b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #pool1
            nnet.addMaxPool(pool_size=(2, 2))

            #conv2
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            
            #pool2
            nnet.addMaxPool(pool_size=(2, 2))

            #conv3
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            
            #pool3
            nnet.addMaxPool(pool_size=(2, 2))

            #conv4
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            
            #pool4
            nnet.addMaxPool(pool_size=(2, 2))

            #conv5
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=256, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            
            #pool5
            nnet.addMaxPool(pool_size=(2, 2))

            #conv6
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=512, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            
            #pool6
            nnet.addMaxPool(pool_size=(2, 2))

            #conv7
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=1024, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            
            #pool7
            nnet.addMaxPool(pool_size=(2, 2))


            # shape is now (2,2) images
            # Code Layer
            nnet.addDenseLayer(num_units=Cfg.dreyeve_rep_dim, b=None)
            nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
            nnet.addReshapeLayer(shape=([0], (Cfg.dreyeve_rep_dim / (2*2)), 2, 2))
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            # unpool1
            nnet.addUpscale(scale_factor=(2, 2))

            # deconv 1
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=1024, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            # unpool2
            nnet.addUpscale(scale_factor=(2, 2))

            # deconv2
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=512, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            # unpool3    
            nnet.addUpscale(scale_factor=(2, 2))

            #deconv3
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=256, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #unpool4
            nnet.addUpscale(scale_factor=(2, 2))

            #deconv4
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #unpool5
            nnet.addUpscale(scale_factor=(2, 2))

            #deconv5
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #unpool6
            nnet.addUpscale(scale_factor=(2, 2))

            #deconv6
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #unpool7
            nnet.addUpscale(scale_factor=(2, 2))

            #deconv7
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            #unpool7
            nnet.addUpscale(scale_factor=(2, 2))

            #deconv8
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=self.channels, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()

            nnet.addSigmoidLayer()
