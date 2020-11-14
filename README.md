# Establish the Business Domain
* what do I want to predict?
* Do I have the right data for it?

<div class="alert alert-warning"><b>NOTE:</b> we are looking at characteristics that are associated with a PERSON making more or less than 50,000 dollars a year. 
### From census website: Some ways data is used:

* provide services for elderly
* build new roads and schools
* locate job training centers

There are various business use cases for this data, from insurance predicting who will likely default on a loan, to understanding the factors - both controllable and uncontrollable - that make someone more or less likely to earn money. 

For this use case, I decided I wanted to investigate factors relevant to the person overall (race, age, immigrant status, etc) so that we can disect the features to understand two things. 

1. What types of people are disadvantaged when it comes to earning money
2. What actions/paths can people take to make their earning outcomes more promisable 

<div class="alert alert-warning"><b>NOTE:</b> We want to be explainable</div>

# Business Assumptions

* I am assuming that the client is more interested in these personal features - normally I would ask the business or domain expert, or ask the client directly. 
* Assuming that although the data is from 1994/1995, they want it to represent the 2020 population. 
* working under the assumption that the working definition of characteristics is a feature that is not obviously related to income, such as: 
    * hourly wage - or if they have hourly wage or not 
    * amount of taxes paid

# Load Data

* Load data
* Load columns

# EDA

<div class="alert alert-warning"><b>NOTE:</b> A lot of my charts are made with plotly, so you cannot see them in display mode, but you can clone the repository, and run the notebooks to interact with these plots. 
</div>

* Quantitative methods of EDA to understand and summarize a dataset, without making any assumptions about what it contains.

* visualizations and tests can be used to help understand what the data looks like, 

* getting familiar with the data, so you can figure out what assumptions you can make that best work with the data and your business problem. After EDA, you can make those assumptions, and then use feature engineering and preprocessing to transform your data to be able to fit those needed assumptions if possible. For example, saying we are assuming balanced data, that doesn't mean that the data is balanced, it just means that the data has the ability to become balanced when using different preprocessing techniques. 

## Steps

### 1. Identify Number of Features or Columns

### 2. Identifying the features or columns

  * Already provided to us in the metadata

### 3. Identifying the data types of features

### 4. Identifying the number of observations

### 5. Checking if the dataset has empty cells or samples

  * No Nan values but a lot of  *?* or  *Not In Universe*

  * What is Not in Universe? [^1]

[^1]: Not  In Universe is a little different from ? because it could be the case that the question doesn't apply to them, but it could also be that they didn't answer the question. If the question didn't apply to them, we could gain some extra value from this *Not in Universe* feature, as the fact that a question doesn't apply to them could give you some indication about the individual. Also, if it just meant that they didn't answer the question, we could gain value from this as well, as there could potentially be a correlation between the number of optional questions someone fills out and their income status. But because we don't know in which case which scenario was present, we will treat these like Nans

### 6. Identifying the number of empty cells by features or columns

### 7. Exploring Categorical features

### 8. Univariate plots, bivariate plots for each feature

* what does relationship mean/what is the significance?
    * boxplot is usually used for count data and histogram is usually used for continuous data, but becuase these continuous features are a littel bit of both, we want to try and use both

    - Age distribution makes sense

    * Sex pretty even - very important
    * A lot of children shown 
    * Weeks worked in year follows a bimodal distribution 

### 9. Bivariate plots

* we can do a million plots for all the combinations of all the features, including using color as a third dimenson. Even though we can automate this task, we don't have time to analyze them all, so we're going to  pick a few important categorical variables that we will look more closely into, although normally, we would want to do this for all combinations of variables.

### 10. Multivariate plots

* Because there is a lot of preprocessing needed to make sense of this data and provide accurate insights, we are going to wait until after data cleaning to visualize this. This is because we don't want to make insights when data is unreliable. For instance, there is a lot of noise that can distract from the insights in these visualizations, for instance including children who obviously will not have any income. 

### 11. Collinarity

* number of people who work for employer and the weeks worked in year are highly positively correlated. This means that although when the value of one goes up, the value of teh other will likely increase. That being said, we can't assume one of these factors causes the other (for instance, working at a bigger company means you work more hours), as there could be an external (or multiple) factors that cause this relationship (for instance, people who work at bigger companies tend to be full time employees (not saying this is true, just an example))


## Additional Questions/Observations from EDA
* What is their definition of income? - e.g. if you are a spouse of someone with an income, does the income here reflect that? 

* A lot of children and family members included - can be a problem if multiple datapoints are within one family because families tend to be more similar, and overrepresentation of larger families can cause the model to learn incorrectly.

* Why are there are people who say they are fully employed, but work 0 weeks in a year

* Why are there are people who say they are fully employed, but work 0 weeks in a year

* columns need to be renamed to follow pep8

* What is migration code and MSA?

  * MSA: An MSA consists of one or more counties that contain a city of 50,000 or more inhabitants, or contain a Census Bureau-defined urbanized area (UA) and have a total population of at least 100,000 (75,000 in New England).

* Income (label) very imbalanced - majority class is making under 50k

* Do the demographics shown reflect real demographics of the time period?
  
  * Want to use *Real-world diversity* - use smapling strategies that increase fairness when trying to support real-world diversity
  
  * Race
  
    * over 83% white - is this reflective of the American population?
    
  * Gender
  
    * Pretty even
  
  * Age
  
    * Follows a distribution that makes sense for age
  


## Assumptions To Make:

* What we have to do to get our data to fit these assumptions 
  
* only want to use people employed full-time because of my assumptions and to not have bias from two people in the same family - if people come from the same household, they will likely have more similar attributes to one another, and therefore their data will be highly correlated which can lead to biases in the algorithm
  
  


# Data Cleaning and Preprocessing
* how to we get our data to work with our chosen assumptions above?
### Features to cut

* cut out variables that you very much think are irrelevant to the business question - features that are directly related to wealth and income
* weight: says to ignore it in the prompt
* wage: directly related to income - it doesn't provide value based on our assumptions about characteristics
* 'capital gains', 'capital losses','dividends from stocks', - too closely related to wealth and therefore will not be able to help with this project's assumed goals

### Replacing Null Values

* Although there are no literal "Nan" values, there are plenty of '?' values in the dataset, which is what this dataset regards as null. In addition, there are plenty of "Not in Universe" datapoints, which as discussed above, we will classify them as null. 

* Want to remove features with over 50% of their data being null - removes a lot of features

  * Because this dataset is pretty large, if we are particularly interested in one feature that has a lot of nulls, for instance state of residence, we could potentially run tests on only rows that have non null values for state of residence. For this use case, I am going to assume that we are not so interested in any of the features with many nulls.

### Dimensionality Reduction
* Don't want to use because we want our model to be explainable so that we can give actionable insights to the stakeholders

### Feature Engineering

* make race look the same way as they do the census now
* Order certain nominal values'
* make immigrant status into a percentage
  * use human development score for country that both themselves and parents are

### Visualizations

* Focus on seeing these categorical variables:
  * sex and education level and how it relates to income 
  * race and education level and how it relates to income 
  * immigrant, education level and how it relates to income

### Transformations 
* planning on using tree based method, where this won't be necessary - this is because tree based methods are not sensitive to the variance of the data 

### Balancing data 

* Using Undersampling - chose for the sake of time. Normally, I would want to try all types of data balancing techniques, like oversampling, undersampling, SMOTE, cluster based over sampling, etc).
* Undersampling is the process where you randomly delete some of the observations from the majority class in order to match the numbers with the minority class. This is done until the majority and minority class instances are balanced out.

* Consider testing under-sampling when you have an a lot data (tens- or hundreds of thousands of instances or more)

  **Advantages**

  - It can help improve run time and storage problems by reducing the number of training data samples when the training data set is huge.

  **Disadvantages**

  - It can discard potentially useful information which could be important for building rule classifiers.
  - The sample chosen by random under sampling may be a biased sample. And it will not be an accurate representative of the population. Thereby, resulting in inaccurate results with the actual test data set.,'

### One Hot Encoding
* need to do this because I want to do tree based model.

### Post Preprocessing Visualizations

* Notice education very important 
* working 52 weeks a year is very densely populated (looking at pairplot) for people who make over 50k
* The population who  make over 50k is more dense at the median (45) compared to those who make less than 50k

# Modeling: Choosing a model

* want the model to perform well, but also be interpretable so that we can understand the features, their importance, how they interact, and how they help determine annual income - don't want to use an unexplainable model like a neural network - want to keep it simple
* Because we have a lot of categorical data - we want to use a tree method.
* Question now is bagging vs boosting - general rule
  * Shallow trees that have high bias low variance - underfitting, use boosting 
  * Deep trees with high variance and low bias - use bagging
* Bagging technique seems to be more fitting with our data, but will try both
  * didn't get great results with random forest, as the precision for the minority class (now balanced) was quite low
  * Trying cat boost - fitting for our data because it works especially well for categorical features - gradient boosting (boosting technique)
  
  <div class="alert alert-warning"><b>IMPORTANT:</b> We want to be explainable</div>

## Tuning My Model

* Cross Validation

* using train, validation, and test set

## Plotting Feature Importance

* Show Feature importance - and the correlations between these important features and our label
* Show tree based model - just showing a shallow tree for vizualization purposes
* You can thus see the most importance features to income through the graph, and can see that feature's correlation to our label to understand how that feature impacts income.