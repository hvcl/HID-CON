
import numpy as np
import os, glob, sys
import tensorflow as tf
#tf.compat.v1.enable_v2_behavior()
from tensorflow import keras
import pickle
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input,Dense, Flatten,  Reshape, Concatenate, MaxPooling2D, GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.layers import Conv2D, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python import debug as tf_debug
from PIL import Image
from tensorflow.keras.constraints import max_norm
import sklearn.metrics 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
tf.get_logger().setLevel("ERROR")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel("WARNING")


from tensorflow.keras import backend as K
import csv, cv2
from datetime import datetime
from skimage.io import imread
from skimage.color import rgba2rgb
from tensorflow.keras.utils import to_categorical
import ast
from skimage.io import imread
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
from tensorflow.keras.regularizers import l1, l2,l1_l2
import ast
from tensorflow import nn
from tensorflow.keras.backend import shape
import pandas as pd
from tensorflow import nn
from tensorflow.keras.applications import VGG16, ResNet50
#from wandb import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tqdm.keras import TqdmCallback
from pathlib import Path

#tf.keras.backend.set_floatx('float16')

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


def _loss_3l_v2(yTrue, yPred):
    output = yPred
    rmse_loss = tf.reduce_mean(tf.square(output-yTrue)) # ok
    sparsity_loss1 = tf.reduce_mean(tf.square(1-tf.reduce_sum(output,axis=1)))
    print ('loss: ',rmse_loss, sparsity_loss1)
    return rmse_loss + sparsity_loss1

def get_output(path,label_file, exp):
    file_name_slide = os.path.split(os.path.split(path)[0])[1]
    with open(label_file, 'r',encoding='utf-8') as inF:
        for line in inF:
            if file_name_slide in line:
                if exp == 'topk':
                    label = int(line[-2])
                else:    
                    label = line[-2].strip()
                    label = ast.literal_eval(label)  
                    label = np.array(label).astype('float16')
                return (label)




class DG(keras.utils.Sequence):
    def __init__(self, list_IDs,label_fn, exp = 'hcd' ,batch_size=32, dim=(256,256), n_channels=3,
                 n_classes=6, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.label_fn = label_fn
        self.exp = exp
        self.on_epoch_end()
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    def __data_generation(self, list_IDs_temp):
        X1 = []
        y = []
        for i, ID in enumerate(list_IDs_temp):
            main = Path(ID)
            X1.append(cv2.imread(str(main))/255)
            output = get_output(main,self.label_fn, self.exp)
            y.append (output)
        return np.array(X1), np.array(y)
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X1, y = self.__data_generation(list_IDs_temp)
        if self.exp == 'topk':
            label = np.zeros((len(y), 2))
            for i in range(len(y)):
                label[i, int(y[i])] = 1
        else:
            label = y
        return X1,label

def main(args):
    time = datetime.now().strftime("%Y%m%d%H%M")
    exp = args.exp
    epochs = args.epochs
    lr = args.lr
    dc = args.dr
    strategy = tf.distribute.MirroredStrategy()

    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    with strategy.scope():
        model = build_model()
        opt = tfa.optimizers.RectifiedAdam(learning_rate=lr, total_steps=1000, min_lr=1e-5,weight_decay=dc)
        model.compile(loss = _loss_3l_v2,optimizer =opt)


    dataset = 'duct_type'#
    model_path = f'warmup_{dataset}_{current_time}_ep30_lr{lr}_dc{dc}_{exp}' 

    ckpt_dir= f"{args.ckpt_dir}{model_path}.h5"
    label_fn = args.label
    data_fn = args.training_set 
    val_fn = args.validation_set

    tr_list = pd.read_csv(data_fn, header=None).to_numpy()[:10]
    tr_list = tr_list[~pd.isnull(tr_list)].tolist()
    #print ("# train: ", len(tr_list))
    val_list = pd.read_csv(val_fn, header=None).to_numpy()[:10]
    val_list = val_list[~pd.isnull(val_list)].tolist()
    bat_num = args.batch_size
    missing = [p for p in tr_list if not Path(p).exists()]



    params = {'shuffle': True, 'batch_size': bat_num, 'exp': exp}
    train_gen= DG(tr_list, label_fn, **params)
    val_gen = DG(val_list, label_fn, **params)

    for ep in range(1, args.epochs):
        print (f"Epoch {ep}")
        history = model.fit(train_gen, epochs = 1,  validation_data = val_gen, verbose = 0,
            workers=8, max_queue_size = 8,use_multiprocessing = True)
        model.save(ckpt_dir) 



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--gpus', type=str, default='0')

    parser.add_argument('--exp', type=str, default='hcd')
    parser.add_argument('--epochs', type=int, default=100, help='Epoch number.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')
    parser.add_argument('--dr', type=float, default=1e-3, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--ckpt_dir', type=str, help='Directory for checkpoint saving.')


    parser.add_argument('--training_set', type=str, help="List of training patches' directory.")
    parser.add_argument('--validation_set', type=str, help="List of validation patches' directory.")
    parser.add_argument('--label', type=str,  help='Slide label list')
 

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpus 
    main(args)


