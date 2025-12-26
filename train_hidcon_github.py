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
from tensorflow.keras.applications import VGG16, ResNet50

from tensorflow.python import debug as tf_debug
import os
#from sklearn.model_selection import KFold 
from PIL import Image
from tensorflow.keras.constraints import max_norm
import sklearn.metrics 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
os.environ["CUDA_VISIBLE_DEVICES"]="3"
tf.get_logger().setLevel("WARNING")
tf.autograph.set_verbosity(2)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import glob as glob
from tensorflow.keras import backend as K
import csv, cv2
import sys
from datetime import datetime
from skimage.io import imread
from skimage.color import rgba2rgb
from tensorflow.keras.utils import to_categorical
import ast
from skimage.io import imread
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l1, l2,l1_l2
from tensorflow.python.keras.layers import Dropout
from tensorflow import nn
from tensorflow.keras.backend import shape
import tensorflow_addons as tfa
import pandas as pd
from collections import Counter


def warmup_model():
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




def _loss2(yTrue, yPred):
    output = tf.reshape(yPred, [-1, 3])
    yTrue = tf.reshape(yTrue, [-1, 3])
    rmse_loss = tf.reduce_mean(tf.square(output - yTrue)) 
    return rmse_loss



def batch_list(ns, batch_size):
    return [ns[i:i+batch_size] for i in range(0, len(ns), batch_size)]


def read_batch_img(batch_paths):
    batch_input = []
    for input_path in batch_paths:
        read_img = cv2.imread(input_path)/255
        batch_input += [read_img]
    batch_x = np.array(batch_input)
    return batch_x


def cluster_mean2(model, patch_fn, intialM, prev_mean=None ):
    print ('------------computing mean for each class----------------')
    with open (patch_fn, 'r',encoding='utf-8-sig') as file_in:
        lines = file_in.readlines()
    lines = [x.split('\n')[0] for x in lines]
    a = batch_list(lines,1024)
    batch_input = []
    for img_batch in a:
        img_arr = read_batch_img(img_batch)
        #print (img_arr.shape)
        if intialM == 1:
            feat = model.predict (np.array(img_arr))
        elif intialM == 0:
            prev_mean1 =  np.repeat(prev_mean[np.newaxis,...], len(img_arr), axis=0)
            feat = model.predict ([img_arr, prev_mean1])
        #print (np.array(feat).shape)
        batch_input += [feat]
    batch_x = np.concatenate(batch_input)
    cluster_mean = np.mean(batch_x,axis= 0)
    #print (len(lines), batch_x.shape, cluster_mean.shape)
    #exit()
    return cluster_mean


def load_mean_file (main_path , path, intialM, model=None, prev_mean=None):
    mean_path = main_path + 'mean_' + os.path.split(path)[1].split('.csv')[0] + '.npy' 
    #print (mean_path)
    if os.path.exists (mean_path) == True:
        mean_file = np.load(mean_path)
    else:
        if intialM == 0:
            mean_file = cluster_mean2 (model, path, intialM, prev_mean=prev_mean)
        else:
            mean_file = cluster_mean2 (model, path, intialM)
        np.save (mean_path, mean_file)
    return mean_file



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
    featsB = Dense(num_class, activation = 'softmax', name = 'output')(featsB)
    inputB = Input(shape=(256,))
    output = Lambda(lambda x: K.stack([x[0], x[1]], axis = 1),name = 'output_con')([featsA, inputB])
    model = Model(inputs=[inputA, inputB], outputs=[output, featsB])
    return model



class SupervisedContrastiveLoss_aug(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss_aug, self).__init__(name=name)
        self.temperature = temperature
    def __call__(self, labels, features, sample_weight=None):
        feature_anchor = features[:,0,:]
        feature_center = features[:,1,:]
        feature_anchor_normalized = tf.math.l2_normalize(feature_anchor, axis=-1)
        feature_center_normalized = tf.math.l2_normalize(feature_center, axis=-1)
        logits = tf.divide(tf.matmul(feature_anchor_normalized, tf.transpose(feature_center_normalized)),
            self.temperature,)
        return (tfa.losses.npairs_loss(tf.squeeze(labels), logits))



def make_pairs(images, labels, aver, true_label):
    pairImages1 = []
    pairLabels1 = []
    pairImages2 = []
    pairLabels2 = []
    numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range (0, numClasses)]
    for idxA in range(len(images)):
        currentImage = images[idxA]
        label = labels[idxA]
        posImage = aver[int(label)]
        pairImages1.append(currentImage)
        pairImages2.append(posImage)
        label_3l = true_label[idxA]
        pairLabels1.append([1])
        pairLabels2.append(label_3l)
        negIdx = np.where(np.unique(labels) != label)[0] 
        Idxn = np.random.choice(np.array(negIdx))
        negImage = aver[Idxn]
        pairImages1.append(currentImage)
        pairImages2.append(negImage)
        pairLabels1.append([0])
        pairLabels2.append(label_3l)
    return ([np.array(pairImages1),np.array(pairImages2)], [np.array(pairLabels1), np.array(pairLabels2)])

class DataGenerator_pairwise_mag(keras.utils.Sequence):
    def __init__(self, list_IDs, labels_file, aver,batch_size=32, dim=(256,256), n_channels=3,
                 n_classes=3, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels_file = labels_file
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.aver = aver
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    def __data_generation(self, list_IDs_temp, list_label_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 1), dtype='float32')
        if self.n_classes == 6:
            z = np.empty((self.batch_size, 6), dtype='float32')
        elif self.n_classes == 3:
            z = np.empty((self.batch_size, 3), dtype='float32')
        for i, ID in enumerate(list_IDs_temp):
            img = imread (ID)
            X[i,] = img/255
            label = list_label_temp[i]
            y[i,] = int(label)
            if label == 0:
                z[i,] = [1,0,0]
            elif label ==1:
                z[i,] = [0,1,0]
            elif label ==2:
                z[i,] = [0,0,1]
        #print (X.shape,y, z)
        return X,y,z
    def __getitem__(self, index):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            list_IDs_temp = [self.list_IDs[k] for k in indexes]
            list_label_temp = [self.labels_file[k] for k in indexes]
            X, y, z = self.__data_generation(list_IDs_temp, list_label_temp)
            (pairTrain1, labelTrain1) = make_pairs(np.array(X), np.array(y).ravel(),self.aver, np.array(z))
            return (pairTrain1, labelTrain1)



def main(args):
    fn = f"{args.ckpt_dir}{args.prev_ckpt}"
    main_path = args.main_dir
    iter = args.prev_ckpt.split('.h')[0]
    tt = '%'
    k = 50
    sdt_path = f'{main_path}top{str(k)}{tt}_class1_tr_{iter}.csv'
    hidden_path = f'{main_path}top{str(int(k/2))}{tt}_hidden_tr_{iter}.csv'
    ldt_path = f'{main_path}top{str(k)}{tt}_class2_tr_{iter}.csv'

    
    
    if args.iter_num == 1: 
        model = warmup_model()
        model.load_weights(fn)
        feat_model = Model(inputs=model.input, outputs= model.get_layer('dense_1').output)
    else:
        model = suconModel2()
        model.load_weights(fn)
        feat_model = Model(inputs=model.input, outputs= model.get_layer('dense_2').output)


    ff = feat_model
    num_class = 3

    
    if args.iter_num == 1: 
        mean_class1 = np.squeeze(load_mean_file(main_path,sdt_path,nn, model=ff ))
        mean_hidden = np.squeeze(load_mean_file(main_path,hidden_path,nn, model=ff ))
        mean_class2 = np.squeeze(load_mean_file(main_path,ldt_path,nn, model=ff ))
        aver = [mean_class1, mean_hidden, mean_class2]   
    else:
        iter_prev = args.prev_prev_ckpt.split('.h')[0]
        kk = '%'
        class1_prev = np.load(f'{main_path}mean_top{(str(k))}{kk}_class1_tr_{iter_prev}.npy')
        hidden_prev = np.load(f'{main_path}mean_top{str(int(k/2))}{kk}_hidden_tr_{iter_prev}.npy')
        class2_prev = np.load(f'{main_path}mean_top{str(k)}{kk}_class2_tr_{iter_prev}.npy')
        prev_mean = class1_prev

        mean_class1 = np.squeeze(load_mean_file(main_path,sdt_path,nn, model=ff, prev_mean=prev_mean ))
        mean_hidden = np.squeeze(load_mean_file(main_path,hidden_path,nn, model=ff, prev_mean=prev_mean ))
        mean_class2 = np.squeeze(load_mean_file(main_path,ldt_path,nn, model=ff, prev_mean=prev_mean ))
        aver = [mean_class1, mean_hidden, mean_class2]
        mean_class1_2 = np.squeeze((mean_class1 + class1_prev)/2)
        mean_hidden2 = np.squeeze((mean_hidden + hidden_prev)/2)
        mean_class2_2 = np.squeeze((mean_class2 + class2_prev)/2)
        aver = [mean_class1_2, mean_hidden2, mean_class2_2]
    



    ## Generate new label for training and validation data
    sdt_path_tr =  sdt_path
    hidden_path_tr = hidden_path
    ldt_path_tr = ldt_path
    y_sdt = np.tile([0], len(pd.read_csv(sdt_path_tr, header=None)))
    y_hidden = np.tile([1], len(pd.read_csv(hidden_path_tr, header=None)))
    y_ldt = np.tile([2], len(pd.read_csv(ldt_path_tr, header=None))) 
    y_all = list(y_sdt) + list(y_hidden) +list(y_ldt)
    sdt_list = pd.read_csv(sdt_path_tr, index_col=None,header = None)
    hidden_list = pd.read_csv(hidden_path_tr, index_col=None, header = None)
    ldt_list = pd.read_csv(ldt_path_tr, index_col=None, header = None)
    train_list = np.array(pd.concat([sdt_list, hidden_list, ldt_list]))
    train_list = [str(x)[1:-1].strip("'") for x in train_list] 

    sdt_path_val =  main_path + f'top{(str(k))}'+ tt +'_class1_val_' + iter + '.csv'
    hidden_path_val = main_path + f'top{str(int(k/2))}' + tt+ '_hidden_val_' + iter + '.csv'
    ldt_path_val = main_path + f'top{(str(k))}' + tt +'_class2_val_' + iter + '.csv'
    y_sdt_val = np.tile([0], len(pd.read_csv(sdt_path_val, header=None)))
    y_hidden_val = np.tile([1], len(pd.read_csv(hidden_path_val, header=None)))
    y_ldt_val = np.tile([2], len(pd.read_csv(ldt_path_val, header=None))) 
    y_all_val = list(y_sdt_val) + list(y_hidden_val) +list(y_ldt_val)
    sdt_list_val = pd.read_csv(sdt_path_val, index_col=None,header = None)
    hidden_list_val = pd.read_csv(hidden_path_val, index_col=None, header = None)
    ldt_list_val = pd.read_csv(ldt_path_val, index_col=None, header = None)
    val_list = np.array(pd.concat([sdt_list_val, hidden_list_val, ldt_list_val]))
    val_list = [str(x)[1:-1].strip("'") for x in val_list]
    

    ## Imbalance Data handling 
    from imblearn.over_sampling import RandomOverSampler
    over_sampler = RandomOverSampler(random_state=42)
    train_list2, y_all2 = over_sampler.fit_resample(np.reshape(train_list,(-1,1)), y_all)
    train_list2 = [str(x)[1:-1].strip("'") for x in train_list2]


    learning_rate = 0.001

    hidden_units = 512
    projection_units = 128



    dropout_rate = 0.5
    temperature = 0.05

    ##Build Model
    initial_learning_rate = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,decay_steps=10000,decay_rate=1e-5,staircase=True) 
    opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    opt = tfa.optimizers.SWA(opt, start_averaging=5, average_period=5) 


    conModel = suconModel2(num_class)
    conModel.compile(optimizer=opt,run_eagerly=True,
    loss= {'output_con':SupervisedContrastiveLoss_aug(temperature = temperature), 'output':_loss2})


    batch_size = args.batch_size

    file_name = f'{iter}_iter{args.iter_num}_t{str(temperature)}_top{k}' 
    time = datetime.now().strftime("%Y%m%d%H%M")


    ckpt_log_dir=f"{args.ckpt_dir}/{file_name}_{time}.h5"
    

    params = {'dim': (256,256), 'batch_size': batch_size,
            'n_classes':num_class, 'n_channels': 3,'shuffle': True}


    training_generator = DataGenerator_pairwise_mag(train_list,y_all , aver, **params)
    validation_generator = DataGenerator_pairwise_mag(val_list, y_all_val,aver, **params)

    for ep in range (args.epochs):
        print ('Trainig epoch ', ep)
        history = conModel.fit(training_generator,epochs=1, verbose =1,validation_data=validation_generator,
                workers= 12 , max_queue_size = 10,use_multiprocessing = True)
        conModel.save(ckpt_log_dir)
        

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--gpus', type=str, default='0')

    parser.add_argument('--prev_ckpt', type=str, help = 'previous checkpoint')
    parser.add_argument('--prev_prev_ckpt', type=str, default='previous preovious checkpoint (previous 2 iterations checkpoint)')
    parser.add_argument('--epochs', type=int, default=100, help='Epoch number.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')
    parser.add_argument('--dr', type=float, default=1e-3, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--iter_num', type=int, default=1, help='Iteration number')
    parser.add_argument('--ckpt_dir', type=str, help='Directory for checkpoint saving.')
    parser.add_argument('--main_dir', type=str, help='Directory for checkpoint saving.')
    parser.add_argument('--label', type=str,  help='Slide label list')   
 

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpus 
    main(args)


