<img src="./images/header.png" alt="header.png" width="1000" >

# Breast Cancer Prediction using Random Forest Classifier

## Using A.I. to Detect Breast Cancer That Doctors Miss
Advancements in A.I. are beginning to deliver breakthroughs in breast cancer screening by detecting the signs that doctors miss. So far, the technology is showing an impressive ability to spot cancer at least as well as human radiologists, according to early results and radiologists, in what is one of the most tangible signs to date of how A.I. can improve public health.

[more information in The NewYork Times](https://www.nytimes.com/2023/03/05/technology/artificial-intelligence-breast-cancer-detection.html)

<br><br>

## Abstract
This project is based on the dataset published by [UCI MACHINE LEARNING](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data) available in Kaggle. The hottest project based on this dataset is developed by [BUDDHINI W](https://www.kaggle.com/code/buddhiniw/breast-cancer-prediction) which has achieved an excellent acccuracy about `94.4%` which looks perfect. However, in this repository, you can gain the accuracy of `99.1%` on test data split.

<br><br>

## Table of Contents
- <a href='#requirements'>Requirements</a>
- <a href='#training-linear-regression'> Training RFC</a>
- <a href='#performance'>Performance</a>
- <a href='#references'>Reference</a>

<br>

## Requirements
This project does not need any specific requirements. The dataset, [UCI MACHINE LEARNING](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data), also is included in the repository.
<br><br>


### Dataset and pre-processing

- We currently provided the dataset in the directory `./dataset/breast-cancer-wisconsin-data/`.  
 
    - Your data directory looks like:
      <br>(Optional) You can change the directory of the dataset when running the `train.py` in terminal.
        ```
        - dataset/
            - __init__.py
            - dataset.py
            - breast-cancer-wisconsin-data/ ...
                - data.csv

        ```

<br>

## Data Mining and Cleaning the Dataset
- There had been a lot of similar works based on this dataset, when I decided to create this repository. The most and foremost difference between this repository and the other repositories published on Github is that i have significantly increased the final accuracy compared to the other developers due to data mining and cleaning the dataset. In this project, i have effectively and efficiently detected and removed all the outliers. Additionally, all the features are now transformed to Normal Distribution.

<br><br>


## Training Random Forest Classifier (RFC)
- We assume that you have cloned the repository.
- To train RFC using the terminal environment, we assume that the dataset is placed in `./dataset/breast-cancer-wisconsin-data/`, so you should simply run the following command. `!python train.py`
By running the following command, you will get more information about the module.

```
>!python train.py -help 
Usage: train.py [options]
Options:
  -h, --help            show this help message and exit
  -p TRAIN_PATH, --path=TRAIN_PATH
                        Path to training data.

```

```
!python train.py -p "./dataset/breast-cancer-wisconsin-data/"
```  
```
!python train.py 

>> The dataset placed in ./dataset/breast-cancer-wisconsin-data/data.csv has been extracted successfully!
>> The dataset comprises 569 various reports based on real cases of breast cancer.
>> 57 outliers are detected and removed.
>> Random Forest Classifier Train Accuracy: 99.76%
>> Random Forest Classifier Test Accuracy: 97.09%
>> 
>> The Best RFC Train Accuracy: 100.00%
>> The Best RFC Test Accuracy: 99.03%
```

<figure>
<img src="./images/skewness.gif" alt="skewness.gif" width="600" >
<figcaption>The Histogram plot corresponding to each features. </figcaption>
<figcaption>Blue Plots: acceptable, Orange Plots: unacceptable </

<br><br>

<img src="./images/skewness_reduction.gif" alt="skewness_reduction.gif" width="600" >
<figcaption>The Histogram plot corresponding to each features after being transformed to Normal Distribution.  </figcaption>
</figure>


<br><br>

<br>

### Colaboratory Notebook
The Second way to train and test RFC is to use the `.ipynb` file in the main directory. It is very informative and builds up your intuition of the process of pre-processing and makes you more knowledgeable about the dataset. In addition, you don't even need to clone the repository, because it can be executed by Google Colaboratory Online.
<br><br>


## Refrences:

The following list contains several links to every resource that helped us implement this project.

1.   Kaggle dataset published by [UCI MACHINE LEARNING](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
2. The project which is developed by [BUDDHINI W](https://www.kaggle.com/code/buddhiniw/breast-cancer-prediction)
3.  Maths-ML developed by [Farzad Yousefi ](https://github.com/F-Yousefi/Maths-ML)
3.  House Price Prediction developed by [Farzad Yousefi ](https://github.com/F-Yousefi/House_Price_Prediction)
4.   Machine Learning course published by [Coursera ](https://www.coursera.org/specializations/machine-learning-introduction)

