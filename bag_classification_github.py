import cv2
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('WARNING')
from tensorflow import keras
from tensorflow.keras.models  import Sequential, Model
from tensorflow.keras.layers import Input,Dense, Flatten, Dropout, Conv2D, MaxPooling2D,BatchNormalization, Reshape, Attention
from tensorflow.keras import regularizers
from PIL import Image
from tensorflow.keras.constraints import max_norm
import numpy as np

os.environ["CUDA_LAUNCH_BLOCKING"]='1'
#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
from skimage.io import imread
from skimage.color import rgba2rgb
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold as KFold
import sklearn.metrics
import ast
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.utils import class_weight
#from imblearn.over_sampling import SMOTE
from tensorflow.keras.regularizers import l1, l2,l1_l2
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Layer

from multiprocessing import set_start_method
set_start_method('spawn')

gpus = tf.config.list_physical_devices('GPU')




def get_output(path,label_file):
    if os.path.split(path)[1][:3] == 'IDH':
        file_name_slide = os.path.split(path)[1].split('.npy')[0][:15]
    elif os.path.split(path)[1][4:6] == 'l1': 
        file_name_slide = os.path.split(path)[1][:6]
    else: 
        file_name_slide = os.path.split(path)[1][:12]
    with open(label_file, 'r',encoding='utf-8') as inF:
        for line in inF:
            if file_name_slide in line:
                label = line[-2]      
                return (label)


class DG_train(keras.utils.Sequence):
    def __init__(self, list_IDs,label_fn, batch_size=1, shuffle=False):
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.label_fn = label_fn
        self.on_epoch_end()
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    def __data_generation(self, list_IDs_temp):
        X = []
        y = []
        for i, ID in enumerate(list_IDs_temp):
            feat = np.squeeze(np.load(str(ID), allow_pickle = True))
            if feat.shape != (100,512):
                feat = np.squeeze(feat)
            feat = np.array(feat, dtype=np.float32)
            label = get_output(ID,self.label_fn)
            X.append(feat)
            y.append(label)
        return np.array(X),np.array(y)
    def __getitem__(self, index):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            list_IDs_temp = [self.list_IDs[k] for k in indexes]
            X, y = self.__data_generation(list_IDs_temp)
            if len (X.shape) ==4:
                X = np.squeeze(X)
            y1 = keras.utils.to_categorical (y, 2)
            return X,y1 



def create_model(input_shape, lr, decay):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = "binary_crossentropy",
                optimizer = tf.keras.optimizers.SGD(lr=lr,decay=decay),# momentum = 0.9), 
                metrics=["accuracy"])
    return(model)    

def main(args):
    day_time = datetime.now().strftime("%Y%m%d%H%M")
    tr_path = f'{args.main_dir}/{args.bag_folder}_conc_tr/'
    img_dir = glob.glob(os.path.join(tr_path, '*'))
    te_path = f'{args.main_dir}/{args.bag_folder}_conc_tr/'
    test_img_dir = glob.glob(os.path.join(te_path, '*.npy'))
    
    
    print ('number of traing sample', len(img_dir))
    print ('number of testing sample', len(test_img_dir))
    input_shape = np.load(img_dir[1]).shape
    print (input_shape)

    input_shape = np.squeeze(np.load(img_dir[0])).shape

    params = {'batch_size': args.batch_size,'shuffle': True}
    lr = args.lr
    dc  = args.dc
    ep = args.epochs
    label_fn = f'{args.main_dir}/{args.label}'
    print ('label directory: ', label_fn)

    yy=0
    df2 = pd.DataFrame(columns=['acc', 'precision', 'recall', 'f1', 'tn' , 'fp', 'fn', 'tp', 'auc'])
    for i in range(args.iter):
        model = None
        model = create_model(input_shape, lr,dc)
        indices = np.arange(np.array(img_dir).shape[0])
        np.random.shuffle(indices)
        shuffled_img_dir = np.array(img_dir)[indices]
        validation_split = 0.2
        num_validation_samples = int(validation_split * np.array(img_dir).shape[0])
        X_train = shuffled_img_dir[:-num_validation_samples]
        X_val = shuffled_img_dir[-num_validation_samples:]
        train_gen = DG_train(X_train,label_fn, **params)
        val_gen = DG_train(X_val,label_fn, **params)
        ckpt_log_dir= f"{args.save_ckpt_dir}/{args.bag_folder}_{lr}_{dc}_gen_iter{i}_ep{ep}.h5"
        history = model.fit(train_gen,epochs =ep, validation_data = val_gen,shuffle = True, verbose = 2)
        index2 = np.arange(len(test_img_dir))
        bsize = args.batch_size
        a = [index2[i:i + bsize] for i in range(0, len(index2), bsize)]
        patch_list =[]
        y = []
        pred =[]
        for batch_ind in a:
            X = []
            for ii in batch_ind:
                path_fn = test_img_dir[ii]
                feat =  np.load(path_fn, allow_pickle=True)
                feat_label =  get_output(path_fn,label_fn)
                X.append(feat)
                y.append(feat_label)
                patch_list.append(path_fn)
            prediction =  model.predict(np.array(X), verbose=0)
            pred.append(prediction)
        y_pred = np.concatenate(pred)
        y_true = np.array(y)
        fn = np.array([os.path.split(i)[1].split('.np')[0] for i in patch_list])
        K.clear_session()
        test_labels = y_true
        score = []
        for o in range(len(y_pred)):
            value = y_pred[o][1]
            score.append (value)
        y_pred2 = np.argmax(y_pred, axis=1)
        sam_la = np.c_[fn,test_labels]
        ff = pd.DataFrame(sam_la)
        ff.columns = ['samples', 'label']
        ff['samples'] = [x[:15] for x in ff['samples']]
        ff = ff.drop_duplicates()
        y_testt = pd.to_numeric(ff.label)
        t = np.c_[fn,y_pred2]
        df = pd.DataFrame(t)
        df.columns = ['samples', 'predicted_labels']
        b = df.groupby(df.samples.str.split('_').str[0]+'_'+ff["samples"].str.split('_').str[1])['predicted_labels'].apply(list)
        b1 =  df.groupby(df.samples.str[:15])['predicted_labels'].apply(list)
        b = b1
        b = b.astype(str).str.replace(r'\[|\]|', '')
        c = b.str.split(', ', expand=True)
        c = c.reset_index()
        c['majority'] = c.mode(axis=1)[0]
        os.makedirs(f'{args.main_dir}/mlp/', exist_ok=True)
        export_csv = c.to_csv (f'{args.main_dir}/mlp/bag_pre_{args.bag_folder}.csv' ) 
        c = pd.read_csv (f'{args.main_dir}/mlp/bag_pre_{args.bag_folder}.csv' ) 
        export_csv = c.to_csv (f'{args.main_dir}/mlp/bag_pre_{args.bag_folder}.csv' ) 
        c2 = c[['samples','majority']]
        c3 = c2.majority.astype(str).str.replace(r'\'|\'|', '')
        f_pred2 = pd.to_numeric(c3.astype(str).str.replace(r'\'|\'|', ''))
        final = ff.merge(c2)
        score_list = pd.DataFrame(np.c_[fn,score])
        score_list.columns = ['samples', 'predicted_labels']
        k = score_list.groupby(score_list.samples.str[:12])['predicted_labels'].apply(list)
        k = k.astype(str).str.replace(r'\[|\]|', '')
        k = k.str.split(', ', expand=True)
        k.reset_index()
        a = pd.to_numeric(final['label']).to_numpy()
        b = pd.to_numeric(final['majority'].astype(str).str.replace(r'\'|\'|', '')).to_numpy()
        acc = sklearn.metrics.accuracy_score(a,b)
        tn, fp, fneg, tp = confusion_matrix(a, b, normalize = 'all').ravel()
        auc = sklearn.metrics.roc_auc_score(to_categorical(a,2),to_categorical(b,2))
        yy=yy+1
        df2.loc[yy, ['acc']] = acc
        df2.loc[yy, ['precision']] = sklearn.metrics.precision_score(a, b,average='weighted')
        df2.loc[yy, ['recall']]= sklearn.metrics.recall_score(a, b,average='weighted')
        df2.loc[yy, ['f1']]= sklearn.metrics.f1_score(a,b,average='weighted')
        df2.loc[yy, ['tn']]= tn
        df2.loc[yy, ['fp']]= fp
        df2.loc[yy, ['fn']]= fneg
        df2.loc[yy, ['tp']]= tp
        df2.loc[yy, ['auc']]= auc
        print (df2.mean(axis=0))
        K.clear_session()




if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--gpus', type=str, default='0')

    parser.add_argument('--main_dir', type=str, help='Directory.')
    parser.add_argument('--save_ckpt_dir', type=str, help='Directory to sasve the checkpoint.')
    parser.add_argument('--bag_folder', type=str,  help='the folder that saved bag features.')   
    parser.add_argument('--label', type=str, help='Label in csv.')
    parser.add_argument('--epochs', type=int, default=200, help='Epoch number.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')
    parser.add_argument('--dc', type=float, default=1e-3, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--iter', type=int, default=10, help='Iteration number')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpus 
    main(args)




