# Salary Prediction - ML Midterm Project, 2024 cohort
## Overview
This is a Classification Machine Learning project completed as part of the certification for the DataTalks Machine Learning Zoomcamp, 2024 cohort. The project aims to predict individuals who earn more than 50,000 dollars per year based on a variety of features. Although the dataset is outdated, it served the purpose of demonstrating our competency in data analysis, preparing data for machine learning operations, and deployment.

The project includes a notebook file, named notebook.ipynb, which covers data cleaning, exploratory data analysis (EDA), preprocessing, model training, and validation. Additionally, there are Python scripts: train.py that encompasses all the necessary steps to train the model, and predict.py, designed for making predictions on new data. Finally, a Dockerfile is included for containerizing the project. The dependencies and packages are managed using Poetry.

### <span style="color:red">**Disclaimer for the evaluation of this project**</span>
We tried to follow the steps from each of the lessons, but we made some modifications because I was looking for an opportunity to experiment with different technologies and steps, and the project turned out to be the ideal opportunity for that. Below are the most important modifications:

1. Instead of Pipenv, Poetry was used here, and all the main files are inside the midterm_project folder. Therefore, instead of `Pipfile` and `Pipfile.lock`, this project has the `pyproject.toml` and `poetry.lock` files.
2. Instead of transforming lists of feature-value mappings to vectors with `DictVectorizer`, we applied different types of encodings using the `pipeline` function from scikit-learn as a way to keep all dataset transformations organized.
3. The entire parameter evaluation and Cross-Validation were performed using a single scikit-learn function called `GridSearchCV`, so we kept the instructions but used different methods.
4. For the web framework, instead of Flask, FastAPI was used, which, along with Swagger UI, made API building and documentation more convenient.

## Dataset
This dataset was found on Kaggle and is called "Salary Prediction Classification". It can be found here: [https://www.kaggle.com/datasets/ayessa/salary-prediction-classification/data](https://www.kaggle.com/datasets/ayessa/salary-prediction-classification/data). The .csv file is located in the [data](./midterm_project/data) folder under the name [salary.csv](./midterm_project/data/salary.csv). Unfortunately, there is no extensive description, but we know that the data was extracted from a 1994 census. The column descriptions only state that the numerical data is continuous and provide the categorical features, but we don't have more information about some of these features:

### Numerical Features
All continuous:
- age
- education-num
- fnlwgt
- capital-gain
- capital-loss
- hours-per-week

### Categorical Features
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

### Target Feature
- salary: <=50K or >50K

## Notebook
**1.0. Data Preparation & Reading**
**2.0. Exploratory Data Analysis**
**3.0. Feature Importance Analysis**
**4.0. Model Selection**
**5.0. Model Saving and Loading**

## Scripts
## Dependencies Files
## Dockerfile
docker build -t midterm-fastapi-app .
docker run -d -p 8000:8000 midterm-fastapi-app
## Deployment
SwaggerUI: http://localhost:8000/docs

### Examples of use
{
  "relationship": "husband",
  "marital_status": "never_married",
  "education": "bachelors",
  "occupation": "prof_specialty",
  "hours_per_week": 40,
  "capital_loss": 0,
  "capital_gain": 0,
  "age": 40,
  "education_num": 13
}

{
  "relationship": "husband",
  "marital_status": "married_civ_spouse",
  "education": "assoc_voc",
  "occupation": "transport_moving",
  "hours_per_week": 50,
  "capital_loss": 0,
  "capital_gain": 5178,
  "age": 55,
  "education_num": 11
}

This is what you need to do for each project

    Think of a problem that's interesting for you and find a dataset for that
    Describe this problem and explain how a model could be used
    Prepare the data and doing EDA, analyze important features
    Train multiple models, tune their performance and select the best model
    Export the notebook into a script
    Put your model into a web service and deploy it locally with Docker
    Bonus points for deploying the service to the cloud

For a project, you repository/folder should contain the following:

    README.md with
        Description of the problem
        Instructions on how to run the project
    Data
        You should either commit the dataset you used or have clear instructions how to download the dataset
    Notebook (suggested name - notebook.ipynb) with
        Data preparation and data cleaning
        EDA, feature importance analysis
        Model selection process and parameter tuning
    Script train.py (suggested name)
        Training the final model
        Saving it to a file (e.g. pickle) or saving it with specialized software (BentoML)
    Script predict.py (suggested name)
        Loading the model
        Serving it via a web service (with Flask or specialized software - BentoML, KServe, etc)
    Files with dependencies
        Pipenv and Pipenv.lock if you use Pipenv
        or equivalents: conda environment file, requirements.txt or pyproject.toml
    Dockerfile for running the service
    Deployment
        URL to the service you deployed or
        Video or image of how you interact with the deployed service