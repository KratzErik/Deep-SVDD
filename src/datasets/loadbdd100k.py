import json
from pathlib import Path, PurePath
import numpy as np
import math
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import time

def load_bdd100k_data_attribute_spec(img_folder, norm_spec, out_spec, n_train, n_val, n_test, out_frac, image_height, image_width, channels, save_name_lists=False, labels_file = None, get_norm_and_out_sets = False, shuffle=False):
    
    # img_folder: path to directory containing BDD100K images
    # norm_spec, out_spec: Specifications of normal and outlier class, respectively. Input are nested lists of string attributes to be included in therespective datasets. Example: norm_spec = [["weather", ["clear","partly cloudy", "overcast"]],["scene", "highway"],["timeofday", "daytime"]]. Not including an attribute, i.e. no spec of weather, will include all weather conditions, i.e. equivalent of specifying all possible keys explicitly.
    # n_train: number of images to be loaded into training set
    # n_val: number of images to be loaded into validation set
    # n_test: number of images to be loaded into testing set
    # out_frac: fraction of outs in testing set
    # image_height, image_width, channels: output data will have these dimensions
    # save_name_lists: if use_file_list=False, file_lists will be created from specified attributes. This options allows to save generated file_lists for reuse with this function
    # labels_file: full path of the JSON file with BDD100K labels.
    # get_norm_and_out_sets: boolean to indicate wether to return normal and outlier data in two sets (True) or train, validation and test set with test labels (False)
    # shuffle: boolean to indicate wether to shuffle data points. Default False => given same attributes and numbers of images, train, val and test sets are identical every time.
  
    n_out_to_choose = int(math.ceil(n_test * out_frac))
    n_norm_test = (n_test - n_out_to_choose)
    n_norm_to_choose = n_train + n_val + n_norm_test
    
    print('Loading json data ...', end='')
    start_time = time.time()
    with open(labels_file) as json_data:
        loaded_json_data = json.load(json_data)
    print('\rLoaded json data (%.2fs)' % (time.time()-start_time))
            
    print('Parsing json data...')
    start_time = time.time()
    norm_filenames= find_matching_files(loaded_json_data, norm_spec, n_norm_to_choose)           
    print('norm file list complete (%.2fs)' % (time.time()-start_time))
        
    start_time = time.time()   
    out_filenames = find_matching_files(loaded_json_data, out_spec, n_out_to_choose)
    print('out file list complete (%.2fs)' % (time.time()-start_time))
        
    if save_name_lists:
        save_file_list(norm_spec, norm_filenames)
        save_file_list(out_spec, out_filenames)
           
    return load_bdd100k_data_filename_list(img_folder, norm_filenames, out_filenames, n_train, n_val, n_test, out_frac, image_height, image_width, channels, get_norm_and_out_sets = get_norm_and_out_sets, shuffle=shuffle)

    
def load_bdd100k_data_filename_list(img_folder, norm_filenames, out_filenames, n_train, n_val, n_test, out_frac, image_height, image_width, channels, get_norm_and_out_sets=False, shuffle=False):
  
    # img_folder: path to directory containing BDD100K images
    # norm_filenames, out_filenames: Lists of strings with names of image files to be included in normal and outlier datasets
    # n_train: number of images to be loaded into training set
    # n_val: number of images to be loaded into validation set
    # n_test: number of images to be loaded into testing set
    # out_frac: fraction of outs in testing set
    # image_height, image_width, channels: output data will have these dimensions
    # save_name_lists: if use_file_list=False, file_lists will be created from specified attributes. This options allows to save generated file_lists for reuse with this function
    # labels_file: full path of the JSON file with BDD100K labels.
    # get_norm_and_out_sets: boolean to indicate wether to return normal and outlier data in two sets (True) or train, validation and test set with test labels (False)
    # shuffle: boolean to indicate wether to shuffle data points. Default False => given same attributes and numbers of images, train, val and test sets are identical every time.
  
    n_out_to_choose = int(math.ceil(n_test * out_frac))
    n_norm_test = (n_test - n_out_to_choose)
    n_norm_to_choose = n_train + n_val + n_norm_test
    
        
    # Assert there is no overlap between target and out data
    overlap_counter = 0
    for filename in out_filenames:
        if filename in norm_filenames:
            overlap_counter += 1
            out_filenames.remove(filename)
            
    if overlap_counter > 0:
        print("\nWARNING: overlap between NORMAL and OUTLIER class: removed %d images from OUTLIER file list\n" % overlap_counter)
        n_out_to_choose -= overlap_counter
        
    # Assert there is enough files to generate sets of requested size
    
    if n_out_to_choose > len(out_filenames):
        print('Not enough files in specified OUTLIER class.\n\tRequested: %d\n\tFound: %d' % (n_out_to_choose, len(out_filenames)))
        scale_factor = len(out_filenames)/n_out_to_choose
        n_out_to_choose = len(out_filenames)
        n_norm_test = int(math.ceil((1 - out_frac)*n_out_to_choose/out_frac))
        n_norm_to_choose = n_train + n_val + n_norm_test
        print('Test set downsized by factor %.2f' % scale_factor)
        
    if n_norm_to_choose > len(norm_filenames):
        print('Not enough files in specified NORMAL class.\n\tRequested: %d\n\tFound: %d' % (n_norm_to_choose, len(norm_filenames)))
        scale_factor = len(norm_filenames)/n_norm_to_choose
        n_norm_to_choose = len(norm_filenames)
        n_train = int(math.ceil(scale_factor * n_train))
        n_val = int(math.ceil(scale_factor * n_val))
        n_norm_test = n_norm_to_choose - n_train - n_val
        # Number of outliers has to be adjusted to keep fraction of outliers constant in test set
        n_out_to_choose = int(math.ceil(out_frac * n_norm_test / (1 - out_frac)))
        print('Train, val and test sets downsized by factor %.2f' % scale_factor) 
        #print("n_train = %d\nn_val = %d\nn_norm_test = %d\nn_out = %d" % (n_train,n_val,n_norm_test,n_out_to_choose))
     
    '''
    if n_out_to_choose > len(out_filenames):
        print('Not enough files in specified OUTLIER class.\n\tRequested: %d\n\tFound: %d' % (n_out_to_choose, len(out_filenames)))
        scale_factor_1 = len(out_filenames)/n_out_to_choose
        n_out_to_choose = len(out_filenames)
        n_norm_test = int(math.ceil((1 - out_frac)*n_out_to_choose/out_frac))
        # n_norm_test reduced means more images can be used for train and val sets
        n_train_and_val = n_train + n_val
        scale_factor_2 = (n_norm_to_choose - n_norm_test)/n_train_and_val
        n_train = int(math.ceil(scale_factor_2 * n_train))
        n_val = n_norm_to_choose - n_train - n_norm_test
        print('Test set downsized by factor %.2f, train and val sets upsized by factor %.2f' % (scale_factor_1,scale_factor_2))
        '''       
        #print("n_train = %d\nn_val = %d\nn_norm_test = %d\nn_out = %d" % (n_train,n_val,n_norm_test,n_out_to_choose))
        
    # Choose image files
    if shuffle:
        norm_perm = np.random.permutation(len(norm_filenames))
        norm_chosen = norm_perm[:n_norm_to_choose]
        
        out_perm = np.random.permutation(len(out_filenames))
        out_chosen = out_perm[:n_out_to_choose]
    else:
        norm_chosen = np.arange(n_norm_to_choose)
        out_chosen = np.arange(n_out_to_choose)
    
    # Keep only selected filenames
    norm_filenames = [norm_filenames[i] for i in norm_chosen]
    out_filenames = [out_filenames[i] for i in out_chosen]
    
    # Specify image format
    norm_data = np.ndarray(shape=(n_norm_to_choose, image_height, image_width, channels), dtype=np.uint8)
    out_data = np.ndarray(shape=(n_out_to_choose, image_height, image_width, channels), dtype=np.uint8)

    # Load norm images
    print("Loading NORMAL image data...")
    start_time = time.time()
    for i, _file in enumerate(norm_filenames):
        img = load_img(str(img_folder / _file))  # this is a PIL image
        img.thumbnail((image_width, image_height))
        x = img_to_array(img)  
        norm_data[i] = x
    print("NORMAL image data loaded (%.2fs)" % (time.time()-start_time))

    # Load out images
    print("Loading OUTLIER image data...")
    start_time = time.time() 
    for i, _file in enumerate(out_filenames):
        img = load_img(str(img_folder / _file))  # this is a PIL image
        img.thumbnail((image_width, image_height))
        x = img_to_array(img)
        out_data[i] = x
    print("OUTLIER image data loaded (%.2fs)" % (time.time()-start_time))

    if get_norm_and_out_sets:
        return norm_data, out_data
    
    else:
        # Divide into train, val and test sets.
        start_time = time.time()

        train_data = norm_data[:n_train] 
        print("Generated train_data (%.2fs)" % (time.time()-start_time))

        start_time = time.time()
        val_data = norm_data[n_train:n_train + n_val]
        print("Generated val_data (%.2fs)" % (time.time()-start_time))

        start_time = time.time()
        test_data = np.concatenate((norm_data[n_train + n_val:], out_data), axis=0)
        print("Generated test_data (%.2fs)" % (time.time()-start_time))

        start_time = time.time()
        test_labels = np.concatenate([np.zeros((len(norm_data[n_train + n_val:])),dtype=int),np.ones((len(out_data),),dtype=int)])
        print("Generated test_labels (%.2fs)" % (time.time()-start_time))
        return train_data, val_data, test_data, test_labels


def find_matching_files(json_data, attributes_to_choose, n_to_choose):
    img_names = []
    
    for entry in json_data:
        add_flag = True
        for attribute in attributes_to_choose:
            if isinstance(attribute[1], list): 
                add_flag = False
                for option in attribute[1]: # if several attribute options are specified, one match is enough => True
                    if entry["attributes"][attribute[0]] == option:
                        add_flag = True
                if not add_flag: # if no option matched, do not evaluate other attributes
                    break
            elif entry["attributes"][attribute[0]] != attribute[1]: # if only one option is specified, any other value => False
                add_flag = False
                break
        if add_flag:
            img_names.append(entry["name"])
        '''       
        check if enough images are found
        if len(img_names) == n_to_choose:
            break
        '''
            
    '''
    scale_factor = -1
    if len(img_names) < n_to_choose:
        attribute_str = ""
        for attribute in attributes_to_choose:
            attribute_str += attribute[0]+": "
            if isinstance(attribute[1], list): 
                for attribute_value in attribute[1]:
                    attribute_str += attribute_value + "/"
            else:
                attribute_str += attribute[1]
            attribute_str += ', '
        err_str = '\tNot enough files with specified attributes (' + attribute_str + ')\n\t\tRequested: %d\n\t\tFound: %d'
        print( err_str % (n_to_choose, len(img_names)))
        scale_factor = len(img_names)/n_to_choose
    '''
        
    return img_names #, scale_factor

def save_file_list(attribute_spec, file_list):
    list_name = ""
    for i1, attribute in enumerate(attribute_spec):
        if i1 > 0:
            list_name += "_and_"
        if isinstance(attribute[1],list):
            for i2, attributeval in enumerate(attribute[1]):
                if i2 > 0:
                    list_name += "_or_"
                list_name += attributeval.replace("/","").replace(" ", "_")
        else:
            list_name += attribute[1].replace("/","").replace(" ", "_")
            
    # Write and save txt file
    with open(list_name+'.txt', 'w') as f:
        for item in file_list:
            f.write("%s\n" % item)
    
