import numpy as np
import os, glob, sys
import tensorflow as tf
from tensorflow import keras
import pickle
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input,Dense, Flatten, Dropout, Reshape, Concatenate, MaxPooling2D, GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.layers import Conv2D, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from PIL import Image
import cv2
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.regularizers import l1, l2,l1_l2
import pandas as pd
import itertools
os.environ["CUDA_VISIBLE_DEVICES"]="4"



def build_model():
    base_model =ResNet50(input_shape= (256, 256,3), weights='imagenet', include_top = False)
    base_model.trainable = True
    x = Flatten()(base_model.output)
    x = tf.keras.layers.Dropout(.5)(x)
    x = Dense(256,activation='relu',bias_regularizer = l1(0.1))(x)
    x = tf.keras.layers.Dropout(.5)(x)
    x = Dense(256, activation='relu', bias_regularizer = l1(0.1))(x)
    x = tf.keras.layers.Dropout(.5)(x)
    output = Dense(3, activation='sigmoid')(x)
    model = Model(inputs = base_model.input, outputs =output)
    return (model)



data_augmentation = tf.keras.Sequential(
    [
        keras.layers.experimental.preprocessing.Normalization(),
        keras.layers.experimental.preprocessing.RandomFlip(),
        keras.layers.experimental.preprocessing.RandomRotation(0.02),
        keras.layers.experimental.preprocessing.RandomContrast(0.05),

    ]
)


def build_hidcon(num_class = 3):
    inputA = Input(shape= (256,256,3))
    base_model = tf.keras.applications.ResNet50(input_shape= (256, 256,3), weights = 'imagenet', include_top = False)
    base_model.trainable = True
    augmented = data_augmentation(inputA)
    feats = base_model(augmented)
    feats = Flatten()(feats)
    feats = Dense(1024,activation='relu')(feats)
    feats = BatchNormalization()(feats)
    feats = Dense(1024,activation='relu')(feats)
    feats = BatchNormalization()(feats)#Dropout(0.5)(feats)
    featsA = Dense(256, activation = 'relu')(feats)
    featsB = Dense(512, activation = 'relu')(feats)
    featsB = Dense(128, activation = 'relu')(featsB)
    featsB = Dense(num_class, activation = 'softmax', name = 'output')(featsB)
    inputB = Input(shape=(256,))
    output = Lambda(lambda x: K.stack([x[0], x[1]], axis = 1),name = 'output_con')([featsA, inputB])
    model = Model(inputs=[inputA, inputB], outputs=[output, featsB])
    return model

def batch_list(ns, batch_size):
    return [ns[i:i+batch_size] for i in range(0, len(ns), batch_size)]


def make_pairs_predict(images):
    pairImages = []
    pairImages1 = []
    proto = np.random.rand(256)
    for idxA in range(len(images)):
        currentImage = images[idxA]
        pairImages.append(currentImage)
        pairImages1.append(proto)
    return [np.array(pairImages),np.array(pairImages1)]



def image_generator_iter0(batch_paths):
    batch_input = []
    batch_input2 = []
    batch_input3 = []
    batch_output = [] 
    for input_path in batch_paths:
        # print ('input_path: ', input_path)
        read_img = cv2.imread(str(input_path))/255
        output = input_path
        batch_input += [read_img]
        batch_output += [output]
    batch_x1= np.array(batch_input)
    batch_y = np.array(batch_output)
    return batch_x1,batch_y



def get_slide_id(path):
    fname = path.split('/')[-2]
    return "-".join(fname.split("-")[:3])


def deploy_iter0 (model, path, new_folder, start=0):
    filename = pd.read_csv(path,header = None )
    filename.columns = [ 'l1']
    paths = filename['l1'].tolist()
    paths_sorted = sorted(paths, key=get_slide_id)
    pred_label = []
    for idx, (name, group) in enumerate(itertools.groupby(paths_sorted, key=get_slide_id)):
        group_list = list(group)
        total_patch = len(group_list)
        file_list = np.array(group_list)
        save_loc = new_folder  + name + '.csv'
        if os.path.exists(save_loc) == True:
            print ('skip', name)
            pass
        else:
            print ('predicting', name)
            img_list = file_list
            a = batch_list(img_list,args.batch_size)
            threelabel = []
            namelist =[]
            for img_batch in a:
                pair_sdt, path = image_generator_iter0 (img_batch)
                pred_3l = model.predict (pair_sdt)
                threelabel.append(pd.DataFrame(pred_3l))
                namelist.append (pd.DataFrame(path))
                K.clear_session()
            all_3l = pd.concat([pd.concat(namelist), pd.concat(threelabel)],axis=1)#
            all_3l.columns = ['l1', 'class1', 'hid', 'class2']
            export_csv = all_3l.to_csv (save_loc, index=None)



def load_mean_file (main_path , path):
    mean_path = main_path + 'mean_' + path + '.npy' 
    print (mean_path)
    mean_file = np.load(mean_path)
    return mean_file


def image_generator(batch_paths):
    batch_input = []
    batch_input2 = []
    batch_input3 = []
    batch_output = [] 
    for input_path in batch_paths:
        read_img = cv2.imread(input_path)/255
        output = input_path
        batch_input += [read_img]
        batch_output += [output]
    batch_x1 = make_pairs_predict(batch_input)
    batch_y = np.array(batch_output)
    return batch_x1, batch_y

def deploy_hidcon(model,  path, new_folder):
    filename = pd.read_csv(path,header = None )
    os.makedirs(new_folder, exist_ok=True)
    pred_label = []
    filename.columns = [ 'l1']
    paths = filename['l1'].tolist()
    paths_sorted = sorted(paths, key=get_slide_id)
    for idx, (name, group) in enumerate(itertools.groupby(paths_sorted, key=get_slide_id)):
            group_list = list(group)
            print(name, len(group_list))
            total_patch = len(group_list)
            file_list = np.array(group_list)
            p= 0
            save_loc = new_folder  + name + '.csv'
            if os.path.exists(save_loc) == True:
                print ('skip', name)
                pass
            else:
                print ('predicting', name)
                a = batch_list(file_list,args.batch_size)
                threelabel = []
                namelist =[]
                for img_batch in a:
                    pair, path = image_generator (img_batch)
                    _,pred_3l = model.predict (pair, verbose=0)
                    threelabel.append(pd.DataFrame(pred_3l))
                    namelist.append (pd.DataFrame(path))
                    K.clear_session()
                all_3l = pd.concat([pd.concat(namelist), pd.concat(threelabel)],axis=1)
                all_3l.columns = ['l1', 'sdt', 'hidden' ,'ldt']
                export_csv = all_3l.to_csv (save_loc, index=None)

def main(args):
    main_dir = args.saving_dir
    ckpt = f'{args.ckpt_dir}{args.ckpt}'
    ckpt_log_dir = os.path.split(ckpt)[1]
    new_folder = f'{main_dir}{args.ckpt.split(".h")[0]}/'
    os.makedirs(new_folder, exist_ok=True)
    print (new_folder)
    if args.task == 'warm_up':
        model = build_model()
        model.load_weights(ckpt)
        deploy_iter0(model,  f'{args.validation_set}', new_folder)
        deploy_iter0(model,  f'{args.validation_set}', new_folder)
    else:
        model = build_hidcon()
        model.load_weights(ckpt)
        deploy_hidcon(model,  f'{args.validation_set}', new_folder)
        deploy_hidcon(model,  f'{args.validation_set}', new_folder) 
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--saving_dir', type=str, help='Directory for saving the predicted label.')
    parser.add_argument('--ckpt_dir', type=str, help='Directory for checkpoint saving.')
    parser.add_argument('--task', type=str, default='warm_up', help='either warm-up stage or the hid-con')
    parser.add_argument('--ckpt', type=str, help='Checkpoint')
    parser.add_argument('--training_set', type=str, help="List of training patches' directory.")
    parser.add_argument('--validation_set', type=str, help="List of validation patches' directory.")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpus 
    main(args)


