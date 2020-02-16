# Machine Learning Skills Test

## Part 1

#### Question 1. Linear Regression using a Neural Network

- In your own words, please describe how you would solve a linear regression problem using a neural network? Describe the network structure, number of layers, layer size(s). You can use symbols/constants/variables to describe how the layer(s) relate to a linear regression problem.

- Now that you have a neural network that can solve linear regression, how would you change this network to do a logistic regression? What would you add/remove from your network?

#### Question 2. Gradient descent

In your own words describe and compare gradient descent, stochastic gradient descent and mini-batch gradient descent. What are the advantages/disadvantages of these?

#### Question 3. Gradient descent optimization

There are many algorithms to optimize gradient descent (eg: Adam, RMSprop, Adagrad, etc.) In your own words describe up-to 3 different gradient descent optimization algorithms. (Please keep your answer short, a general idea about these is enough)

#### Question 4. Softmax function

Would you use a softmax function in the final layer of a multi-class multi-label classification problem? What are the advantages/disadvantages of doing this?

#### Question 5. Prior expertise

Briefly describe, in your opinion, the most interesting/challenging project you have worked on (preferably related to machine learning). Describe what made this project interesting/challenging and how you overcame any issues that arose during this project. Also mention if you worked in a team or individually and what role you had if it was a team project.


## Part 2

You are provided with a file `data.csv` that contains the number of bicycles observed in several places in Ottawa during 2010 to 2019. The csv file has the following columns:

| column name | description |
|:-------------:|:-------------:|
| location_name | the location where the counter was installed |
| count      | number of bicycles passed by |
| max temp | maximum temperature (Celsius) |
| mean temp | average temperature (Celsius) |
| min temp | minimum temperature (Celsius) |
| snow on grnd (cm) | snow on ground |
| total precip (mm) | total precipitation |
| total rain (mm) | total rain |
| total snow (cm) | total snow |
| date | date of recording |


This part requires you to perform **data engineering**, **classical machine learning** methods and **neural networks** methods to solve a multi-class classification problem. All code should be written in Python (within the provided Jupyter Lab environment) with or without (but not restricted to) the following packages:

- `pandas`
- `numpy`
- `sklearn`
- `scipy`
- `tensorflow`
- `pytorch`

If you wish to use any other dependencies, please specify them in a `requirements.txt` file. All code should be run on a CPU machine.

**Task objective:** given an input of `date, max temp, mean temp, min temp, snow on grnd (cm), total precip (mm), total rain (mm), total snow (cm)`, predict whether the total number of bicycles observed in a day is ***less than 2000***, ***in between 2000 and 10k*** or ***over 10k***.

Please read through all the items below before you start the task.

#### 1. Pre-processing

```python
import pandas as pd

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Pre-process a dataframe

    :param pd.DataFrame df: raw dataframe from data.csv

    :returns pd.DataFrame processed_df: processed dataframe
    '''
    ...
    return processed_df
```

Please implement the above function which does the following two steps:

1. First sum the counts at different locations in a day. In other words, you are creating a new `DataFrame` that contains the following columns:

- `date`
- `max temp`
- `mean temp`
- `min temp`
- `snow on grnd (cm)`
- `total precip (mm)`
- `total rain (mm)`
- `total snow (cm)`
- `total count`

2. Convert the numerical column `total count` into three categories: *less than 2000*, *2000 to 10000* and *over 10000*. Save the `processed_df` to `processed_data.csv`.

#### 2. Data engineering

```python
def data_engineering(processed_df: pd.DataFrame) -> (pd.DataFrame,
                                                     pd.DataFrame):
    '''
    Perform data engineering on processed dataframe

    :param pd.DataFrame processed_df: output of preprocess()

    :returns pd.DataFrame train_df: training set of the engineered dataframe
    :returns pd.DataFrame test_df: test set of the engineered dataframe
    '''
    ....
    return (train_df, test_df)
```

Please implement the above function to do any data engineering as needed, split the dataset into train set (80%) and test set (20%). The final `DataFrame` should be ready for training/testing, in other words, one should be able to get `x, y` arrays using the following commands:

```python
x_train = train_df.drop(columns=['total count']).values
y_train = train_df['total count'].values
```

Save `train_df`, `test_df` to `train.csv` and `test.csv` respectively.

#### 3. Classical machine learning methods

```python
def classical_ml(train_df: pd.DataFrame, test_df: pd.DataFrame) -> (
    'classifier', 'accuracy', 'confusion matrix'):
    '''
    Use classical machine learning methods to predict total counts

    :param pd.DataFrame train_df: training set dataframe
    :param pd.DataFrame test_df: test set dataframe

    :returns 'classifier': trained classifier
    :returns 'accuracy': tuple of training accuracy and testing accuracy
    :returns 'confusion matrix': confusion matrix on test set
    '''
    x_train = train_df.drop(columns=['total count']).values
    y_train = train_df['total count'].values
    x_test = test_df.drop(columns=['total count']).values
    y_test = test_df['total count'].values

    clf = model.fit(x_train, y_train, ...)
    ...
    return (clf, (train_acc, test_acc), confusion_matrix)
```

Please complete the above function which takes `train_df` and `test_df` as inputs, and outputs the trained classifier, accuracy on training set and test set, confusion matrix on test set. You can experiment with as many methods as you want, please only leave the one with best performance (you can leave the others in comments) so that the function only returns one trained classifier.

Please answer the following questions:

1. What data engineering techniques did you apply?
2. What's the best accuracy on test set did you achieve? Which classifier did you use to get the best accuracy?
3. Which features are the most important for predicting counts of bicycles?

#### 4. Neural networks

```python
def nn_ml(train_df: pd.DataFrame, test: pd.DataFrame) ->  ('model',
                                                           'test_accuracy'):
    '''
    Use neural networks to predict total counts

    :param pd.DataFrame train_df: training set dataframe
    :param pd.DataFrame test_df: test set dataframe

    :returns 'model': trained model
    :returns 'test_accuracy': accuracy on test set
    '''
    x_train = train_df.drop(columns=['total count']).values
    y_train = train_df['total count'].values
    x_test = test_df.drop(columns=['total count']).values
    y_test = test_df['total count'].values

    mdl.fit(x_train, y_train, ...)
    ...

    return (mdl, test_acc)
```

Please complete the above function which takes `train_df` and `test_df` as inputs, and outputs the trained model, accuracy on test set. You can experiment with as many structures as you want, please only leave the one with the best performance so that the function only returns one model. In addition to the model and test accuracy, please also produce two graphs:

- accuracy on train and validation data, `x-axis`: epochs, `y-axis`: accuracy
- loss on train and validation data, `x-axis`: epochs, `y-axis`: loss

Note validation data is not test set, it should be split from training set.

Please answer the following questions:

1. How many epochs did you train? How did you decide when to stop training?
2. Please briefly explain the model structure (layers, sizes) you choose.


#### 5. Discussions

Please write your observations, comments regarding this dataset and problem, you can also tell us about what challenges you faced or what you have learnt during your experiments.


#### 6. Optional

If you are to design a forecasting model using this dataset to predict the counts of bicycles (still three categories, not actual numbers) in the future, what changes will you make? Please state all factors that you consider relevant, there is no need to write code for this question.
