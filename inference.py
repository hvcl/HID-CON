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



data_augmentation = keras.Sequential(
    [
        keras.layers.experimental.preprocessing.Normalization(),
        keras.layers.experimental.preprocessing.RandomFlip(),
        keras.layers.experimental.preprocessing.RandomRotation(0.02),
        keras.layers.experimental.preprocessing.RandomContrast(0.05),

    ]
)

def suconModel2(num_class = 3):
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
    featsB = Dense(num_class, activation = 'softmax', name = 'output_6l')(featsB)
    inputB = Input(shape=(256,))
    output = Lambda(lambda x: K.stack([x[0], x[1]], axis = 1),name = 'output_con')([featsA, inputB])#tf.keras.layers.Concatenate(axis=0,name = 'output_con')([featsA, inputB])
    model = Model(inputs=[inputA, inputB], outputs=[output, featsB])
    return model


def batch_list(ns, batch_size):
    return [ns[i:i+batch_size] for i in range(0, len(ns), batch_size)]


def make_pairs_predict(images, aver):
    pairImages = []
    pairImages1 = []
    pairImages2 = []
    pairImages3 = []
    for idxA in range(len(images)):
        currentImage = images[idxA]
        pairImages.append(currentImage)
        pairImages1.append(aver[0])
        pairImages2.append(aver[1])
        pairImages3.append(aver[2])
    return ([np.array(pairImages),np.array(pairImages1)],[np.array(pairImages), np.array(pairImages2)],[np.array(pairImages),np.array(pairImages3)])


def image_generator(batch_paths, aver):
    batch_input = []
    batch_input2 = []
    batch_input3 = []
    batch_output = [] 
    for input_path in batch_paths:
        #print (input_path)
        read_img = cv2.imread(input_path)/255
        output = input_path
        batch_input += [read_img]
        batch_output += [output]
    batch_x1, batch_x2, batch_x3 = make_pairs_predict(batch_input, aver)
    batch_y = np.array(batch_output)
    #print (batch_y.shape)
    return batch_x1,batch_x2, batch_x3, batch_y

def make_pairs_predict(images, aver):
    pairImages = []
    pairImages1 = []
    pairImages2 = []
    pairImages3 = []
    for idxA in range(len(images)):
        currentImage = images[idxA]
        pairImages.append(currentImage)
        pairImages1.append(aver[0])
        pairImages2.append(aver[1])
        pairImages3.append(aver[2])
    return ([np.array(pairImages),np.array(pairImages1)],[np.array(pairImages), np.array(pairImages2)],[np.array(pairImages),np.array(pairImages3)])


def image_generator(batch_paths, aver):
    # print ('image_generator')
    batch_input = []
    batch_input2 = []
    batch_input3 = []
    batch_output = [] 
    for input_path in batch_paths:
        #print (input_path)
        read_img = cv2.imread(input_path)/255
        output = input_path
        batch_input += [read_img]
        batch_output += [output]
    batch_x1, batch_x2, batch_x3 = make_pairs_predict(batch_input, aver)
    batch_y = np.array(batch_output)
    # print (np.array(batch_x1[0]).shape, np.array(batch_x1[1]).shape)
    # print (np.array(batch_x2[0]).shape, np.array(batch_x2[1]).shape)
    # print (np.array(batch_x3[0]).shape, np.array(batch_x3[1]).shape)
    return batch_x1,batch_x2, batch_x3, batch_y



def deploy(model,  path, new_folder, aver, start=0):
    filename = pd.read_csv(path,header = None )
    filename.columns = [ 'l1']
    filename = filename.sort_values(by=['l1']) 
    os.makedirs(new_folder, exist_ok=True)
    pred_label = []
    for i, (name, group) in enumerate(itertools.islice(filename.groupby(filename.l1.str[27:42]), start, None), start=start):
            total_patch = len(group)
            file_list = np.array(group)
            save_loc = new_folder  + name + '.csv'
            if os.path.exists(save_loc) == True:
                print ('skip', i,name)
                pass
            else:
                print ('predicting', name)
                img_list = [str(x)[1:-1].strip("'") for x in file_list] 
                a = batch_list(img_list,64)
                sdt = []
                hidden = []
                ldt = []
                threelabel = []
                namelist =[]
                for img_batch in a:
                    pair_sdt, pair_hidden, pair_ldt, path = image_generator (img_batch, aver)
                    _,pred_3l = model.predict (pair_sdt, verbose=0)
                    threelabel.append(pd.DataFrame(pred_3l))
                    namelist.append (pd.DataFrame(path))
                    K.clear_session()
                all_3l = pd.concat([pd.concat(namelist), pd.concat(threelabel)],axis=1)#
                all_3l.columns = ['l1', 'sdt','hid', 'ldt']
                export_csv = all_3l.to_csv (save_loc, index=None)


def main(args):

    #model=tf.keras.models.load_model(args.model_path, compile=False)
    model = suconModel2()
    model.load_weights(args.model_path)
    print ('Done loading trained model')
    os.makedirs(args.save_path, exist_ok=True)
    aver = np.load(args.aver_path)

    deploy(model, args.input_file, args.save_path, aver, start=0)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='HID-CON Args')
    parser.add_argument('--model_path', type=str,   default='.../ckpt.hdf5')
    parser.add_argument('--input_file', type=str, default='.../patch_list.csv', help='CSV file that contains list of patches.')
    parser.add_argument('--save_path', type=str, default='.../save_path/')
    parser.add_argument('--aver_path', type=str, default='.../aver_path/aver.npy')
    args = parser.parse_args()
    main(args)

