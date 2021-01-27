
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
classify_model = octid.octid(model_name = 'GoogleNet', model=None, dim = 3, SVM_nu = 0.03, templates_path = 'templates_path', val_path = 'val_path', unknown_path='unknown_path')

# run the classify model
classify_model()

# parameters
# model_name: you can use [pretrained torchvision models](https://pytorch.org/docs/stable/torchvision/models.html)
# model: or use your own model
# dim: feature dimension after using Umap, we recommend setting is to 3 
# SVM_nu: we are using the rbf kernel for SVM. This parameter is an upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Should be in the interval (0, 1]. By default 0.03 will be taken.
# templates_path: the path of your template dataset folder, which should only contain the positive(cancerous) images.
# val_path: the path of your validation dataset folder, which should contain both positive and negative images.
# unknown_path: the path of the dataset that you want to classify, which will be divided into two categories and placed in two folders after running our classify model

#Dataset folders notes: since we are using the [torchvision.datasets.ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) to label the image, please follow the way to creat your image folders. And the image should be cut dowm to small images such as 500 by 500, not the original medical micro image.
```

## Contributing 

### Setup (based on [this](https://scikit-learn.org/stable/developers/contributing.html#contributing-code))
If you're a developer and wish to contribute, please follow these steps:

1. Create an account on GitHub if you do not already have one.

2. Fork the project repository: click on the ‘Fork’ button near the top of the page. This creates a copy of the code under your account on the GitHub user account. For more details on how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

3. Clone your fork of the hyperopt repo from your GitHub account to your local disk:

   ```bash
   $ git clone https://github.com/<github username>/hyperopt.git
   $ cd hyperopt
   ```

### Setup a python 3.x environment for dependencies
4. Create environment with:  
   `$ python3 -m venv my_env` or `$ python -m venv my_env`
   or with conda:  
   `$ conda create -n my_env python=3`

5. Activate the environment:  
   `$ source my_env/bin/activate`  
   or with conda:  
   `$ conda activate my_env`

6. Install dependencies for extras (you'll need these to run pytest):
   Linux/UNIX:
   `$ pip install -e '.[MongoTrials, SparkTrials, ATPE, dev]'`

   or Windows:
   ```cmd
   pip install -e .[MongoTrials]
   pip install -e .[SparkTrials]
   pip install -e .[ATPE]
   pip install -e .[dev]
   ```

7. Add the upstream remote. This saves a reference to the main hyperopt repository, which you can use to keep your repository synchronized with the latest changes:

    `$ git remote add upstream https://github.com/hyperopt/hyperopt.git`

    You should now have a working installation of hyperopt, and your git repository properly configured. The next steps now describe the process of modifying code and submitting a PR:

8. Synchronize your master branch with the upstream master branch:

    ```bash
    $ git checkout master
    $ git pull upstream master
    ```

9. Create a feature branch to hold your development changes:

    `$ git checkout -b my_feature`

    and start making changes. Always use a feature branch. It’s good practice to never work on the master branch!


### Formatting
10. We recommend to use [Black](https://github.com/psf/black) to format your code before submitting a PR which is installed automatically in step 4.

11. Then, once you commit ensure that git hooks are activated (Pycharm for example has the option to omit them). This will run black automatically on all files you modified, failing if there are any files requiring to be blacked. In case black does not run execute the following:

    ```bash
    $ black {source_file_or_directory}
    ```

12. Develop the feature on your feature branch on your computer, using Git to do the version control. When you’re done editing, add changed files using git add and then git commit:

    ```bash
    $ git add modified_files
    $ git commit -m "my first hyperopt commit"
    ```

### Running tests
13. The tests for this project use [PyTest](https://docs.pytest.org/en/latest/) and can be run by calling `pytest`.

14. Record your changes in Git, then push the changes to your GitHub account with:

    `$ git push -u origin my_feature`

Note that dev dependencies require python 3.6+.


## Algorithms

Currently three algorithms are implemented in hyperopt:

- [Random Search](http://www.jmlr.org/papers/v13/bergstra12a.html?source=post_page---------------------------)
- [Tree of Parzen Estimators (TPE)](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
- [Adaptive TPE](https://www.electricbrain.io/blog/learning-to-optimize)

Hyperopt has been designed to accommodate Bayesian optimization algorithms based on Gaussian processes and regression trees, but these are not currently implemented.

All algorithms can be parallelized in two ways, using:

- [Apache Spark](https://spark.apache.org/)
- [MongoDB](https://mongodb.com)

## Documentation

[Hyperopt documentation can be found here](http://hyperopt.github.io/hyperopt), but is partly still hosted on the wiki. Here are some quick links to the most relevant pages:

- [Basic tutorial](https://github.com/hyperopt/hyperopt/wiki/FMin)
- [Installation notes](https://github.com/hyperopt/hyperopt/wiki/Installation-Notes)
- [Using mongodb](https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB)

## Related Projects

* [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn)
* [hyperopt-nnet](https://github.com/hyperopt/hyperopt-nnet)
* [hyperas](https://github.com/maxpumperla/hyperas)
* [hyperopt-convent](https://github.com/hyperopt/hyperopt-convnet)
* [hyperopt-gpsmbo](https://github.com/hyperopt/hyperopt-gpsmbo/blob/master/hp_gpsmbo/hpsuggest.py)

## Examples

See [projects using hyperopt](https://github.com/hyperopt/hyperopt/wiki/Hyperopt-in-Other-Projects) on the wiki.

## Announcements mailing list

[Announcements](https://groups.google.com/forum/#!forum/hyperopt-announce)

## Discussion mailing list

[Discussion](https://groups.google.com/forum/#!forum/hyperopt-discuss)

## Cite

If you use this software for research, please cite the paper (http://proceedings.mlr.press/v28/bergstra13.pdf) as follows:

Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. TProc. of the 30th International Conference on Machine Learning (ICML 2013), June 2013, pp. I-115 to I-23.

## Thanks

This project has received support from

- National Science Foundation (IIS-0963668),
- Banting Postdoctoral Fellowship program,
- National Science and Engineering Research Council of Canada (NSERC),
- D-Wave Systems, Inc.
