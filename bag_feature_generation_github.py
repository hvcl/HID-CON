import numpy as np
from skimage.io import imread
from skimage.color import rgba2rgb
import sys, os, argparse, cv2

import pandas as pd
import skimage.io
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, Dense
#from keras.utils import print_summary
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


from tensorflow.keras.applications.vgg16 import preprocess_input
from skimage.transform import rescale, resize



def load_sample_images( out_dir,a ):
    samples = {}
    fd = open( out_dir + a + '.csv' ) 
    for line in fd:
        line = line.split(',')
        #print (line)
        samples[line[0]] = [ fn.strip() for fn in line[1:] if fn.strip() != '' ]
    return samples


def load_extractor(img_shape = (256,256,3)):
    model_name = 'vgg'
    layer = 'block5_pool'#'
    pool_size = 8
    base_model = tf.keras.applications.VGG16(input_shape=img_shape,include_top=False,weights='imagenet')
    x = base_model.get_layer(layer).output
    if pool_size is None:
        x = GlobalAveragePooling2D(name='avgpool')(x)
    elif pool_size is not None:
        p = int(pool_size)
        if p > 0:
            x = MaxPooling2D((p,p),name='maxpool')(x)
    model2 = Model( inputs=base_model.input, outputs=x )
    model2.summary()

    return model2

def main(args):
    extractor = load_extractor()
    split_file = args.split_file

    sample_images = load_sample_images( args.main_dir, args.bag_list)
    lev = args.rawpatch_dir.split('L')[1]


    sample_list = []

    for sample,imagelist in sample_images.items():
        sample_list.append(sample)

    print (sample_list[1])

    folder_fn = 'feature_'+ args.bag_list.split('sample_images_')[1].split('_50B_500p')[0] + args.model_name 
    npy_dir = args.main_dir+ folder_fn+'/'
    tefolder_fn = folder_fn + '_conc/'
    trfolder_fn = folder_fn + '_conc_tr/'
    os.makedirs(args.main_dir +  trfolder_fn, exist_ok=True,mode=0o0777  )
    os.makedirs(args.main_dir+  tefolder_fn, exist_ok=True,mode=0o0777 )

    start = int(len(sample_list)//9*6)
    end = int(len(sample_list)//9*9)


    for sample,imagelist in sample_images.items():
            if sample in sample_list[start:end] :
                print('Extracting ==> ',sample)
                loc = args.main_dir + trfolder_fn +str(sample)+'.npy'
                loc1 = args.main_dir + tefolder_fn +str(sample)+'.npy'
                a = os.path.exists(loc) 
                b = os.path.exists(loc1)
                batch_input = []
                if a or b == True:
                    print (sample,'existing')
                    pass
                else: 
                    for img_fn in imagelist:
                            folder_fn2 = img_fn[:15]
                            src_dir = args.rawpatch_dir
                            p1 = src_dir  + folder_fn2 + '/' + img_fn
                            feat_fn = npy_dir+img_fn[:img_fn.rfind('.')]+ '.npy' 
                            img =cv2.imread(p1)
                            x = (img)/255
                            batch_input += [x]
                            p = extractor.predict(np.array(batch_input), verbose=0)
                            if len(p.shape) > 2:
                                feat = [ p[:,r,c,:].squeeze() for r in range(p.shape[1]) for c in range(p.shape[2]) ]
                            else:
                                feat = [ p.squeeze() ]
                            b =  folder_fn2
                            fold = pd.read_csv (split_file,index_col=None)
                            label = np.array(fold[fold['ID'] == folder_fn2]['label'])
                            if label == 'train':
                                np.save (loc,feat)
                            else:
                                np.save (loc1, feat)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--gpus', type=str, default='0')

    parser.add_argument('--main_dir', type=str, help='Directory.')
    parser.add_argument('--bag_list', type=str,  help='the csv file which contain the bag list')   
    parser.add_argument('--rawpatch_dir', type=str, help='Raw patch directory.')
    parser.add_argument('--model_name', type=str, default='vgg')
    parser.add_argument('--split_file', type=str, default='', help='csv file that indicates a slide is a train or test sample. ')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpus 
    main(args)