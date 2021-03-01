# DisasterResponsePipeline

This project is part of Udacity's Data Scientist Nanodegree program. There are two given datasets, one provides social media or newspaper messages of people in need and the other dataset provides the corresponding category of help needed.

These two datasets are processed with an ETL pipeline and afterwards the resulting databse is used for generating a machine learning model. This model is used for a webapp, where you can classify messages in disaster categories. 

### Installation

There are some files in the repository which are stored in Git Large File System (Git LFS), e.g. the database or the machine learning model.

If you have never used Git LFS before, it is required to download and install Git LFS - [click here for more infomration](https://git-lfs.github.com/). After a succesfull installation you need to set up Git LFS with the following command:

```
git lfs install
```
Then you can clone the repository as usual with following command:
```
# using HTTPS
git clone https://github.com/lethiess/DisasterResponsePipeline.git
```
or
```
# using SSH
git clone git@github.com:lethiess/DisasterResponsePipeline.git
```

### Prerequisites

All parts of this project are based on **Python 3**. There are some additinal requirements for the ETL and ML Pipeline and the webapp. You need to install the following python packages:
- flask
- numpy
- pandas
- sqlalchemy 
- plotly
- scikit-learn

#### Anaconda

It is recommended to use the [Anaconda](https://anaconda.org/) python distribution. You can create an environment with all necessary packages with the following command:

```
conda env create -f environment.yml
```

### Run the ETL Pipeline

Open a terminal in the ```data``` directory and run:
```
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.DB
```

Alternatively you can use VS Code and the provided launchfile for a convenient workflow [(more information)](#vscode).

### Run the ML Pipeline

Open a terminal in the ```models``` directory and run:
```
python train_classifier.py ../data/DisasterResponse.DB DisasterResponseModel.pkl
```

Alternatively you can use VS Code and the provided launchfile for a convenient workflow [(more information)](#vscode).

### Run the webapp

Open a terminal in the ```app``` directory and run:
 ```
 python run.py 
 ```

After a successful server startup, open a browser and type in following address: [http://127.0.0.1:3001/](http://127.0.0.1:3001/).

### Working with Visual Studio Code <a name="vscode"></a>

It is recommended to work with Microsoft Visual Studio Code (VS Code). If you open the repositories main folder with VS Code there is a lauchfile for running the ETL am ML pipeline with the correct input arguments. 
