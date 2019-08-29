Experiment-3
==============================

Almost done! We have completed Line Text predictor. Now comes the part of implementing Line Detector. For this, we will use IAM dataset again but paragraph dataset.

The objective in this experiment is to design a line detector. Given a paragraph image the model must be able to detect each line. What do you mean by detect? We will preprocess the paragraph dataset such that each pixel corresponds to either of the 3 classes i.e. 0 if it belongs to background, 1 if it belongs to odd numbered line and 2 if it belongs to even numbered line.

Now that we have dataset, images with paragraph of size (256, 256) and ground truths of size (256, 256, 3) we use full convolution neural networks to give output of size (256, 256, 3) for an input of (256, 256). We use 3 architectures, lenet-FCN (converted to FCNN), resnet-FCN and custom-FCN.

Results are bit embarassing.

Refer the [blog](https://dudeperf3ct.github.io/project/2019/05/17/Fun-in-Deep-Learning-Project/#experiment-3) for further details.

**Learnings**

- Investigate as to why model is not performing well in segmenting. Having a good line segmentor is critical for our OCR pipeline.



A short description of the project.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    |
    |
    └── src                 <- Source code for use in this project
        ├── data
        │   ├── dataset.py
        │   ├── iam_dataset.py   <- Scripts to download or generate data
        │   ├── iam_paragraphs.py
        |
        ├── __init__.py            <- Makes src a Python module
        |
        ├── models
        │   ├── base_model.py
        │   ├── line_detect_model.py
        |
        ├── networks
        │   ├── custom_fcn.py
        │   ├── fcn.py
        │   ├── lenet_fcn.py
        │   ├── resnet.py
        |
        ├── tests                    <- Scripts to use trained models to make predictions
        │   ├── support
        ├── training                        <- Scripts to train models
        │   ├── clr_callback.py
        │   ├── lr_find.py
        │   ├── train_model.py                 
        │   └── util.py
        ├── util.py
        └── visualization       <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
