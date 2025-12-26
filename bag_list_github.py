import os
import glob
import pandas as pd
import numpy as np
import csv
from itertools import groupby
from operator import itemgetter
from collections import defaultdict
from tqdm import tqdm


def main(agrs):
    main_dir = args.main_dir
    columns = []
    columns = defaultdict(list)
    list_dir = main_dir + args.patch_list
    list_= np.concatenate(pd.read_csv(list_dir).to_numpy())


    print("image list: ", len(list_))
    data = []
    k1 =[]
    k2 = []
    k3 = []
    label = []
    data2 = []


    maxOfBag = args.num_bag
    patchNum = args.num_patch
    t=0
    for p in tqdm(range(maxOfBag), desc = 'Buiding bag ...'):
        res = [list(i) for j, i in groupby(list_, lambda a: os.path.split(os.path.split(a)[0])[1])]
        for i in range (len(res)):
            group = res[i]
            k = os.path.split(os.path.split(group[0])[0])[1]
            b = len(group)
            batch_paths = np.random.choice(a = group, size = patchNum, replace=True)
            gr = [os.path.split(x)[1].split('\n')[0].split(',')[0]  for x in batch_paths]
            gr1 = pd.DataFrame (batch_paths)
            t = t+1
            ww = gr1.iloc[:, 0].tolist()
            data2.append (gr)
            data.append (gr1)
            new_k = str(k)+'_'+ str(p+1)
            k2.append (new_k)
            k1.append (k)



    fn2 = os.path.split(list_dir)[1].split('.csv')[0]
    df1 = pd.DataFrame(data2)
    df2 = pd.DataFrame(k2)


    df_con = pd.concat([df2, df1], axis=1)
    print  ('Number of samples: ' , len(df_con))
    export_csv = df_con.to_csv (main_dir +'sample_images_'+ fn2 + '_' + str(maxOfBag) +'b'+str(patchNum)+ 'p.csv', index=None, header = False) 


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Args')

    parser.add_argument('--main_dir', type=str, help='Directory for checkpoint saving.')
    parser.add_argument('--patch_list', type=str,  help='the csv file with filtered patch list')   
    parser.add_argument('--num_bag', type=int, default=50, help='Number of bag.')
    parser.add_argument('--num_patch', type=int, default=100, help='Number of patch per bag.')

    args = parser.parse_args()
    main(args)