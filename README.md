https://www.kaggle.com/datasets/ayessa/salary-prediction-classification/data


- age: continuous.
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: continuous.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
- salary: <=50K or >50K




DOCKER

docker build -t midterm-fastapi-app .
docker run -d -p 8000:8000 midterm-fastapi-app

SWAGGERUI
http://localhost:8000/docs

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
} -> 0

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
} -> 1













Starting with Poetry

New project: poetry new <name of the project>
Just create a .toml file: poetry init
Shell: poetry shell
Deactivate: deactivate
List of Environments: poetry env list
Delete: delete
Set my environment to local folder: poetry config virtualenvs.in-project true

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