import pandas as pd
import numpy as np
import glob , os

def main(args):
    fn = args.saving_folder 
    main_dir = args.saving_dir
    filenames = glob.glob(os.path.join(fn  ,'*.csv'))
    print (f"Number of slides: {len(filenames)}")
    label_fn = args.set_label
    class1_top50 =[]
    class2_top50 =[]
    hid_top50 =[]
    df = []
    i=0
    for j in range(len(filenames)):
        input_path = filenames[j]
        slide_fn = os.path.split(input_path)[1].split('.csv')[0]
        print (slide_fn)
        with open(label_fn, 'r',encoding='utf-8') as inF:
            for line in inF:
                if slide_fn in line:
                    data = pd.read_csv(input_path)
                    #print (data)
                    data.columns=['filename', 'class1', 'hid', 'class2']
                    filename = data.filename
                    pred = data.drop('filename', axis=1)
                    pred = np.array(pred)
                    pred_label = [ np.argmin(x) for x in pred]
                    #print ("before: ", slide_fn, len(filename))
                    num_top50_hid = int(len(filename)*0.25)
                    num_top50 = int(len(filename)*0.5)
                    top50_hid = data.nlargest(num_top50_hid, columns='hid')
                    hid_top50.append(top50_hid['filename'])
                    i = i+1
                    with open(label_fn, 'r',encoding='utf-8') as inF:
                        for line in inF:
                            if slide_fn in line:
                                label = line[16].strip() 
                                if label == '0':
                                    idx = np.where(np.array(pred_label) == int(label))
                                    top50_class1 = data.nlargest(num_top50, columns='class1')
                                    class1_top50.append(top50_class1['filename'])
                                    #print (i, 'sdt',slide_fn, len(top50_class1))
                                elif label == '1':
                                    idx = np.where(np.array(pred_label) == int(label))
                                    top50_class2 = data.nlargest(num_top50, columns='class2')
                                    class2_top50.append(top50_class2['filename'])
                                    #print (i,'ldt',slide_fn, len(top50_class2))
                                img_list = np.array(filename)[idx]
                                df.append(pd.DataFrame(img_list))
            

    all_df = pd.concat (df)
    hid_top50 = pd.concat (hid_top50)
    class2_top50 = pd.concat (class2_top50)
    class1_top50 = pd.concat (class1_top50)
    save_fn =os.path.split(args.saving_folder)[1]
    
    set_fn = args.set
    tt = '%'
    save_dir = main_dir
    hid_top50.to_csv (save_dir + 'top25'+tt +'_hidden_' +set_fn +'_' + save_fn + '.csv', index=None, header = False) 
    class1_top50.to_csv (save_dir + 'top50'+tt +'_class1_' +set_fn + '_' + save_fn + '.csv', index=None, header = False) 
    class2_top50.to_csv (save_dir + 'top50'+tt +'_class2_'+ set_fn +'_' + save_fn + '.csv', index=None, header = False) 



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--saving_dir', type=str, help='Directory for filtered patch list.')
    parser.add_argument('--saving_folder', type=str, help='Directory for saving the predicted label.')
    parser.add_argument('--ckpt_dir', type=str, help='Directory for checkpoint saving.')
    parser.add_argument('--set', type=str, default = 'val',  help='train or validation')
    parser.add_argument('--set_label', type=str, help='Groundtruth label for each set (e.g.: train or validation).')
    args = parser.parse_args()
    main(args)



