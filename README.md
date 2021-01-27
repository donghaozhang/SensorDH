
# OCTID: One-Class learning-based tool for Tumor Image Detection

OCTID is a novel Python package, which utilizes a pre-trained CNN model, UMAP,and one-class SVM for cancerous image detection based on the partially annotated dataset.

## Getting started

Install hyperopt from PyPI

```bash
$ pip install octid
```

to run your first example

```python
from octid import octid
# initialize the classify model with the requiered parameters
classify_model = octid.octid(model_name = 'GoogleNet', model=None, dim = 3, SVM_nu = 0.03, 
                              templates_path = 'templates_path', val_path = 'val_path', unknown_path='unknown_path')

# run the classify model
classify_model()

# parameters
# model_name: you can use [pretrained torchvision models](https://pytorch.org/docs/stable/torchvision/models.html)
# model: or use your own model
# dim: feature dimension after using Umap, we recommend setting is to 3 
# SVM_nu: we are using the rbf kernel for SVM. This parameter is an upper bound on the fraction of training 
          #errors and a lower bound of the fraction of support vectors. Should be in the interval (0, 1]. By default 0.03 will be taken.
# templates_path: the path of your template dataset folder, which should only contain the positive(cancerous) images.
# val_path: the path of your validation dataset folder, which should contain both positive and negative images.
# unknown_path: the path of the dataset that you want to classify, which will be divided into two categories 
                and placed in two folders after running our classify model

#Dataset folders notes: since we are using the [torchvision.datasets.ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) to label the image, please follow the way to creat your image folders. And the image should be cut dowm to small images such as 500 by 500, not the original medical micro image.
```

### Setup a python3 environment for octid
1. Create environment with conda:  
   `$ conda create -n my_env python=3`

2. Activate the environment with conda:  
   `$ conda activate my_env`

3. Install required package:
   install_requires=[
        "torch",
        "torchvision",
        "itertools",
        "matplotlib",
        "numpy",
        "sklearn",
        "pandas",
        "os",
        "random",
        "shutil",
        "umap"]


