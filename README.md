# Establish the Business Domain

* what do I want to predict?
* Do I have the right data for it?

<div class="alert alert-warning"><b>NOTE:</b> we are looking at characteristics that are associated with a PERSON making more or less than 50,000 dollars a year. 

### From census website: Some ways Data is used:

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

<div class="alert alert-info"><b>NOTE:</b> I wanted to display as little code as possible on the notebooks to be as interpretable to everyone, regardless our their techincal knowledge. Because of this, I added a python file in the nbs folders called notebook_functions.py that has all the more technical coding. Please refer to this file if you are interested in learning how I used Python to solve this challenge. 
</div>

