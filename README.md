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
   
```python
python inference.py --input_file xxx.csv
```
