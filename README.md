# Diabetes Prediction

## Why?

Someone asked me for help with predicting Diabetes using a dataset, and I later found that it's the [Kaggle WiDS Datathon 2021](https://www.kaggle.com/c/widsdatathon2021).

This was my first major foray into Data Science and Machine Learning, and I had a lot of fun!
I'm sharing this so that I can refer to it later, and so that it can help others getting started.

I'm leaving a basic outline of my process in the Readme, but you should refer to the attached [iPython Notebook](WIDS2021.ipynb) if you have the time.

* [Preprocessing](#pre-processing)
  - Remove features with too much missing data.
  - Encode category columns with LabelEncoder.
  - Impute missing values with IterativeImputer.
  - Remove outliers with IsolationForest.
* [Feature Engineering](#feature-engineering)
  - Generate new features
  - Remove correlated features.
* [Training](#training)
  - Estimate number of iterations required for LGBM.
  - Train LGBM.

## Pre-processing

You can find details about the dataset and the metrics for evaluation at the Kaggle competetion linked above.
The gist is, there are about 180 feature columns, and a Target column with boolean values indicating if the patient has Diabetes Mellitus or not.There were 130157 samples of labelled data, and 10234 samples of unlabelled data.

Most of the samples have missing values. Of the 130k samples in labelled data, only about 3000 had no missing values.
I dropped feature columns with too many missing values (more than 30,000 in labelled), and columns which are unlikely to be
good indicators of Diabetes.

There are some categorical features like Ethnicity, which I encoded using `sklearn.preprocessing.LabelEncoder`.
In hindsight, I should have left the categorical features [to lgbm](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#categorical-feature-support).

Then I imputed the missing values in the dataset using `sklearn.impute.IterativeImputer`.

Since the dimensionality of the dataset is high, I cannot use my usual, naive approach to outlier detection: Distance from mean.
After some Google-fu, I found out IsolationForests work well with high-dimensionality datasets.
`sklearn.ensemble.solationForest` to the rescue!

## Feature Engineering

I generated new feature columns by binning bmi, height, weight, and age.

Then I generated several new features like:
- Difference between max and min columns of a lab measurement. Example: d1_glucose_max, and d1_glucose_min.
- Is the Daily maximum and Hourly maximum of a lab measurement same.
- Is the Daily minimum and Hourly minimum of a lab measurement same.
- Difference of a feature's value from the mean of the feature.
- Difference of a feature's value from the mean of the feature for other people in the same BMI/Height/Weight/Age bin.
- Difference of a feature's value from the mean of the feature for other Diabetes Positive people in the same BMI/Height/Weight/Age bin.
- Difference of a feature's value from the mean of the feature for other Diabetes Negative people in the same BMI/Height/Weight/Age bin.

Then I removed highly correlated features.
I used the SULA method from [auto-viml](https://github.com/AutoViML/Auto_ViML) to find correlated features.
You can find it in the [uncorr.py](uncorr.py) file.

## Training

I estimated Number of Iterations to be used for LGBM by cross validation as suggested in https://sites.google.com/view/lauraepp/parameters.

Then I created an ensemble of 5 LGBM classifiers which predict the probability of Diabetes, and combined them with equal weight.

Here are my parameters for LGBM:
- Boosting: Goss
- Metric: Area Under RoC curve.
- Learning Rate: 0.01
- Number of Iterations: about 7000.


## Results

My AuC RoC was 0.86791. There were around 800 teams, and the winning team has an AuC of 0.87804.

The difference in AuC is about 0.01, but my rank was 170 (at the time of writing).
Here is what I would do differently if I tried to close the gap:

- Generate more features
- Put back features the correlation finder dropped, but (I think) should stay.
- Pass the indexes of categorical features to LGBM so that it can process them better.
- Actually learn how to properly [tune hyperparameters for LGBM](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html).
- Train more ensembles with other classification algorithms (catboost?).

One could probably push the 0.01 with these, but it would take a [lot more time and effort](https://www.google.com/search?q=pareto+principle) than I'm willing to spend.
I had already spent the better part of a week on this, and the tiny improvement would not have an impact meaningful enough to warrant indulging in it more.


