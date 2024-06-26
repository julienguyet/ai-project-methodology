# Churn Prediction
==============================

A hands on project on churn prediction to discover the best practice of a machine learning project (code versioning, reproducibility, documentation, tracking with ML Flow, etc.) by Malika Matissa, Preetha Pallavi, and Julien Guyet. 

## 1. Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── Dataset       <- Where to store complete dataset to be used for prediction job.
    │
    ├── docs               <- A Sphinx project to see code documentation. Open the index.html file to see it:
                            ai-project-methodology/docs/build/html/index.html
    │
    ├── mlflow             <- Contains metadata and artifacts for MLflow experiments, providing a structured 
                              approach to manage and track machine learning experiments.
        ├── mlruns         <- Stores logged parameters, metrics, and run metadata.
        ├── package_code   <- Contains the machine learning project organized as an MLflow project.
    │
    ├── models             <- Trained and serialized models: encoder and model joblib files needed to run the 
                              predictions on fresh data.
    │
    ├── notebooks          <- Jupyter notebooks. Execute Ecommerce-final.ipynb to run the prediction job.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment. You can
    │                         install them with `pip install -r requirements.txt`
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## 2. Environment set up
--------

To reproduce the environment run the following command in order:

```
conda create --name ai-methodo
```

```
pip install -r requirements.txt
```

You now have all required libraries to execute the code from this repo. 

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
