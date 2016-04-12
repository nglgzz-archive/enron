# Project 5: Identify Fraud from Enron Email


## Introduction
Enron, one of the biggest energy companies in the world, was an American energy,
commodities and services company that was worth $111 billion during 2000, but
in the end of 2001 after the revelation of willful corporate fraud and
corruption it went bankrupt.

In this project I used Enron public dataset containing financial information
and emails of some of the company's employees to build a machine learning model
to identify employees that could have committed fraud.

Machine learning is useful in solving this kind of problems, because it makes
possible to quickly and automatically build a model to make accurate
predictions. This is a classification task, which means we have
some features as input, and from those the model is gonna predict the class
of membership of that input, the classes in this case are: poi and non-poi,
with poi standing for person of interest, meaning that they most likely have
committed fraud.


## Dataset
The dataset contains 146 elements of which two are outliers, that leaves us
with 144 persons, among these, 18 (12.5%) are persons of interest while 126
(87.5%) are not persons of interest. This imbalance between classes, if not
dealt properly with, can cause problems on a classification task, and I will
address how I dealt with it later on the report.

Between the different features available there are financial information like
salary, bonus and stock value; and email information like total number of
emails sent and received and total number of emails sent or received from a
poi. The corpus of the emails is also available in the dataset, but I didn't
use it on this project.

One of the two outliers I found contained the sum of all the financial data,
while the other one had a negative `'total_stock_value'`. I removed them both
because the first one was not meant to be there, while the second one didn't
make sense, since stocks can't have a negative value.

The dataset contained many NaN values, which I then converted to -1 to be sure
they were counted in the learning process but at the same time that they
weren't treated as other numbers.
```
{   'bonus': 64,
    'deferral_payments': 107,
    'deferred_income': 97,
    'director_fees': 129,
    'email_address': 35,
    'exercised_stock_options': 44,
    'expenses': 51,
    'from_messages': 60,
    'from_poi_to_this_person': 60,
    'from_this_person_to_poi': 60,
    'loan_advances': 142,
    'long_term_incentive': 80,
    'other': 53,
    'poi': 0,
    'restricted_stock': 36,
    'restricted_stock_deferred': 128,
    'salary': 51,
    'shared_receipt_with_poi': 60,
    'to_messages': 60,
    'total_payments': 21,
    'total_stock_value': 20}
```

## Features
Among all the financial features available I chose `'salary'`, `'bonus'` and
`'total_stock_value'`; while for email features I created `'from_poi_perc'`
and `'to_poi_perc'`, which respectively are the percentage of emails received
by a person from a poi and the percentage of emails sent by a person to a poi.
I used those features because I think they can best represent the importance
and the relationship of an employee with a poi, and I created the two email
features instead of using the gien ones, because a relative number in this
case can be better than the absolute values.


When selecting from the features, I used SelectKBest (and chi squared as score
function) and after testing different `k` values ("all", 4, 3) I realized that
in this case keeping all the features would give the best F1 score.
Below are the values I tried and the scores I got.
```
# k = "all"
Accuracy:  0.837
Precision:  0.492  | Recall:  0.865
F1:  0.599

# k = 4
Accuracy:  0.793
Precision:  0.428  | Recall:  0.880
F1:  0.549

# k = 3
Accuracy:  0.680
Precision:  0.236  | Recall:  0.580
F1:  0.324
```
And the p-values for each feature.
```
[   ('salary', '0.072'),
    ('bonus', '0.034'),
    ('total_stock_value', '0.016'),
    ('from_poi_perc', '0.385'),
    ('to_poi_perc', '0.272')]
```

I also had to scale the features since chi squared doesn't accept negative
values, and the converted NaN values were all set to -1. I used MinMaxScaler
for this purpose, which scales the values in a range between 0. and 1.


## Algorithm
There are many algorithms that can be used on a classification algorithm, and
there isn't an algorithm that it's always the best, it depends from the
dataset. The algorithms I tried are: support vector machine, nearest neighbor,
naive bayes and decision tree.
Below are the scores (I cut some of the output for clarity).
```
SVC(...)
Accuracy:  0.867
Precision:  0.000  | Recall:  0.000
F1:  0.000

KNeighborsClassifier(...)
Accuracy:  0.859
Precision:  0.168  | Recall:  0.120
F1:  0.133

GaussianNB()
Accuracy:  0.859
Precision:  0.369  | Recall:  0.320
F1:  0.326

DecisionTreeClassifier(...)
Accuracy:  0.785
Precision:  0.360  | Recall:  0.675
F1:  0.455
```
After testing and evaluating the four algorithms, I picked decision tree for
my project. Even though at first it could seem the worst performing algorithm
because it has the lowest accuracy, it is actually the best one in terms of
finding persons of interest. The problem is as I mentioned before that the
classes are really unbalanced so if for example we had a model that predicted
non-poi for all the employees it would still have 87.5% (non-poi/total)
accuracy, but such model wouldn't be very useful. For this reason I used other
metrics to measure performance, which are: precision, recall and F1.


## Tuning
Parameter tuning is a process in which parameters of a
function are tweaked in order to achieve a performance improvement and it
is important because depending on the kind of problem and on the data you
have, some parameter may work well and some others may work really bad; for
instance when I tried changing the decision tree parameter `class_weight` from
`"balanced"` to `None` it dropped both precision and recall almost to 0.0.

In this phase I tried instantiating the algorithm using
different parameters, at each parameter change I tested the algorithm and I
kept the parameters that were giving the best performance.
While doing this it is important to have a consistent validation and evaluation
method, otherwise it could lead to choose the wrong parameters and being
misled by untrue or imprecise scores; an example could be, if I measured
the performance of my model using just the accuracy and base my tuning on that,
I could end up having an algorithm that classifies everyone as non-poi and
think that it is indeed good, when it actually is pretty useless.

I tried different values for the min_samples_split parameter (25, 50 and 100)
and I also tried two different values for class_weight (None, "balanced"),
below are the scores (I cut some of the output for clarity).
```
# min_samples_split = 100, class_weight = None
Accuracy:  0.862
Precision:  0.002  | Recall:  0.005
F1:  0.003

# min_samples_split = 25, class_weight = "balanced"
Accuracy:  0.837
Precision:  0.492  | Recall:  0.865
F1:  0.599

# min_samples_split = 50, class_weight = "balanced"
Accuracy:  0.771
Precision:  0.405  | Recall:  0.895
F1:  0.528

# min_samples_split = 100, class_weight = "balanced"
Accuracy:  0.777
Precision:  0.388  | Recall:  0.650
F1:  0.456
```

## Validation
Validation is a way to assess how a model generalizes what it has learned to
new data. It is generally done by slicing the data into training and testing
sets (or some variation of that), with the first one being the data used for
the actual learning, while the second one is used for evaluation.
Here I used a StratifiedShuffleSplit which splits the data into training and
testing sets several times, mantaining the same percentage of classes as in the
original dataset, I then train and test the classifier several times and
return the mean of the scores as evaluation.

A classic mistake that can be done is using the same data for both testing and
training, in that case the algorithm would most likely get good scores, but
in reality it could be prone to overfitting or simply not be really good.


## Evaluation
When evaluating the predictions of a model there are many different metrics
available, and depending on the problem and on the dataset, one tries to
maximize the score of one metric rather than another. For instance here the
accuracy, which is the ratio between the number of right predictions and the
total number of predictions, as I just explained is not really useful. I
therefore used precision, recall and F1 as metrics for evaluation. The average
score of my model in these metrics is: precision = 0.431, recall = 0.828,
F1 = 0.567; which means, 43.1% of the times a poi is classified, it is a real
poi, 82.8% of the poi are classified as poi, and F1 is the harmonic mean
between precision and recall.


## Sources
- [Enron (Wikipedia)](https://en.wikipedia.org/wiki/Enron)
- [Machine Learning (Wikipedia)](https://en.wikipedia.org/wiki/Machine_learning)
- [Scikit Learn Documentation](http://scikit-learn.org/)

I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc.