# HID-CON: Weakly Supervised Intrahepatic Cholangiocarcinoma Subtype Classification of Whole Slide Images using Contrastive Hidden Class Detection

<p align="center">
  <img src="hiddenClass_mil_231026.jpg"  >
</p>


# Framework 
Tensorflow 2

# Preprocessing
1. Download the raw WSI data.
2. Prepare the patches.
3. Store all the patches directory in a .csv file.

# Download Checkpoint
The checkpoint can be downloaded from [here](https://huggingface.co/jingwei92/HID-CON/tree/main).

# Inference
   The 'best_aver.npy' should be first downlaoded from this page first.
```python
 python inference_github.py --model_path .../best.hdf5 --input_file ../xx.csv --save_path /xxx/xxx --aver_path .../best_aver.npy
```
