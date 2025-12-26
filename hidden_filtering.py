import pandas as pd
import numpy as np
import glob , os

def main(args):
    filenames = sorted(glob.glob(f"{args.main_dir}{args.saving_folder}/*"))
    patch_list = []
    for file in filenames: 
        df = pd.read_csv(file)
        df.columns = ['l1', 'class1', 'hid', 'class2']
        max_pred5_rows = df[df['hid'] == df[['class1', 'hid', 'class2']].max(axis=1)]
        df2 = df.drop(max_pred5_rows.index)['l1']
        print (os.path.split(file)[1], len(df), len(df2))
        patch_list.append (df2)
    patch_list2 = pd.concat(patch_list)
    patch_list2.to_csv(f'{args.main_dir}{args.saving_folder}_filtered.csv', index=None)



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--main_dir', type=str, help='Directory for filtered patch list.')
    parser.add_argument('--saving_folder', type=str, help='Directory for saving the predicted label.')
    args = parser.parse_args()
    main(args)
