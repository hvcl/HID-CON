## HID-CON: Weakly Supervised Intrahepatic Cholangiocarcinoma Subtype Classification of Whole Slide Images using Contrastive Hidden Class Detection 
By [Jing Wei Tan](https://scholar.google.com/citations?user=_PMI46gAAAAJ&hl=en),  Kyoungbun Lee and  [Won-Ki Jeong](https://scholar.google.com/citations?user=bnyKqkwAAAAJ&hl=en&oi=sra)

[Paper](https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-12/issue-6/061402/HID-CON--weakly-supervised-intrahepatic-cholangiocarcinoma-subtype-classification-of/10.1117/1.JMI.12.6.061402.full)
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
