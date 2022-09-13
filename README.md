# <a name="top"></a>Zillow Single Family Residence Value Predictor
![]()

by: Jennifer M Lewis

<p>
  <a href="https://github.com/JenniferMLewis" target="_blank">
    <img alt="Jennifer" src="https://img.shields.io/github/followers/JenniferMLewis?label=Follow Jenn&style=social" />
  </a>
</p>


***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___

<img src="https://images.unsplash.com/photo-1480074568708-e7b720bb3f09?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2348&q=80">

## <a name="project_description"></a>Project Description:
[[Back to top](#top)]

"You are a junior data scientist on the Zillow data science team and recieve the following email in your inbox:

    We want to be able to predict the property tax assessed values ('taxvaluedollarcnt') of Single Family Properties that had a transaction during 2017.

    We have a model already, but we are hoping your insights can help us improve it. I need recommendations on a way to make a better model. Maybe you will create a new feature out of existing ones that works better, try a non-linear regression algorithm, or try to create a different model for each county. Whatever you find that works (or doesn't work) will be useful. Given you have just joined our team, we are excited to see your outside perspective.

    One last thing, Maggie lost the email that told us where these properties were located. Ugh, Maggie :-/. Because property taxes are assessed at the county level, we would like to know what states and counties these are located in.

-- The Zillow Data Science Team"
***
## <a name="planning"></a>Project Planning: 
[[Back to top](#top)]

### Project Outline:
Identify an improved model for assessing value of single family properties for Zillow, using Zillow data about single family houses sold in 2017.
- Which features have the most weight on price?
- Does Area of the home, or number of Rooms play more of a role in house price?
        
### Hypothesis
- Area (sqft) of the house will have the higest affect of the variables we have.
- More Bedrooms have more value than More Bathrooms.


### Target variable
- House Value is the target which is the tax_value variable

### Need to haves (Deliverables):
- README [this file]
- Final Notebook [with model above baseline accuracy]
- Python Files for Exploration


### Nice to haves (With more time):
- Analysis of affect of location on price of homes
- Analysis of affect of property size on price of homes
- More statistical tests [this feels like my weakpoint]


***

## <a name="findings"></a>Key Findings:
[[Back to top](#top)]

- Bedrooms have less bearing on value than originally expected.
- Bathrooms play more of a role in the higher priced houses.
- Area, as expected, plays the largest role in house price.
- Bedrooms are only more valuable when it comes to the lower end of the array.


***

## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]

### Data Used
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| bathroomcnt | Number of bathrooms in home including fractional bathrooms | float64 |
| bedroomcnt | Number of bedrooms in home | float64 |
| calculatedfinishedsquarefeet | Calcuated total finished living area of the home | float64 |
| taxvaluedollarcnt | The total tax assessed value of the parcel | float64 |
| transactiondate | Date that the property had a transaction | object |

---
#### Property locations using FIPS 
###### (Don't worry Maggie, it's not lost anymore!)

|FIPS |State |County |
| ----- | ----- | ----- |
| 06037 | CA | Los Angeles |
| 06059 | CA | Orange |
| 06111 | CA | Ventura |

---
***

## <a name="wrangle"></a>Data Acquisition and Preparation
[[Back to top](#top)]

![]()


### Wrangle steps: 

- Check if there is a CSV file named zillow_2017_mvp.csv for the data and loads it.
- If not, pulls bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, and transactiondate from properties_2017, propertylandusetype, and predictions_2017, sorted where the propertylandusedesc is "Single Family Residential", and saves as zillow_2017_mvp.csv
- Now the dataframe has columns renamed 'bedroomcnt' to'bedrooms', 'bathroomcnt' to 'bathrooms', 'calculatedfinishedsquarefeet' to 'area', and 'taxvaluedollarcnt' to 'tax_value'.
- Then any properties with 0 bedrooms are dropped, they aren't much of a single family residence if there's no place to sleep.
- Next outliers in the bedroom, bathroom, area, and tax_value columns are removed.
- After the transactiondates were verified to all be in 2017.
- Then a histogram and box plot are created with the dataframe
- The date column is dropped since all properties are 2017 in line with the requested data for analysis.
- The dataframe is then split into Train, Validate, and Test.
- Train, Validate, and Test are returned to the user.


*********************

## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:
    - wrangle.py 
    - explore.py
    - evaluate.py


### Takeaways from exploration:
- Wrangle:
    - The date column was probably not actually needed because the removal of outliers dropped the row without a 2017 date.
    - Removing outliers dropped all the rows will null or missing values, making cleaning a breeze.
- Explore:
    - With the variables we have, there is only vague correlations, there is definitely better variables to be discovered, time permitting.
    - Combinations of Area, Beds, and Baths can help narrow the error margin slightly.
- Evaluate:
    - Bedrooms play less of a role than expected
    - Whereas, Bathrooms play a larger role than expected. 

***

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]


### Stats Test 1: T-Test: One Sample, Two Tailed
- A T-test allows one to compare a categorical and a continuous variable by comparing the mean of the continuous variable by subgroups based on the categorical variable
- The t-test returns the t-statistic and the p-value:
    - t-statistic: 
        - Is the ratio of the departure of the estimated value of a parameter from its hypothesized value to its standard error. It is used in hypothesis testing via Student's t-test. 
        - It is used in a t-test to determine if you should support or reject the null hypothesis
        - t-statistic of 0 = H<sub>0</sub>
    -  - the p-value:
        - The probability of obtaining test results at least as extreme as the results actually observed, under the assumption that the null hypothesis is correct
- We wanted to compare the individual clusters to the total population. 
    - Cluster1 to the mean of ALL clusters
    - Cluster2 to the mean of ALL clusters, etc.

#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is No difference between means.
- The alternate hypothesis (H<sub>1</sub>) is Area mean's tax_value is higher than the mean of tax_vaue.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05


#### Results:
We can Reject the Null Hypothesis.
This suggests there is more value in homes with above the median area than homes with median and below area.


#### Summary:
This was repeated for Bathroom number mean value, and Bedroom number mean value. All results showed the value for Bathroom and Bedrooms were higher than the average tax_value. Same as showed with Home Area.

### Stats Test 2: Independent T-Test: Two Sample, Two Tailed
- A T-test allows me to compare a two continuous variable by comparing the mean of the continuous variables
- The t-test returns the t-statistic and the p-value:
    - t-statistic: 
        - Is the ratio of the departure of the estimated value of a parameter from its hypothesized value to its standard error. It is used in hypothesis testing via Student's t-test. 
        - It is used in a t-test to determine if you should support or reject the null hypothesis
        - t-statistic of 0 = H<sub>0</sub>
    -  - the p-value:
        - The probability of obtaining test results at least as extreme as the results actually observed, under the assumption that the null hypothesis is correct
- We wanted to compare the individual clusters to other individual clusters. 
    - Cluster1 to the mean of Cluster2.

#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is average value for Area/Bathrooms/Bedrooms have no difference.
- The alternate hypothesis (H<sub>1</sub>) is there is higher value in one of the variables.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05


#### Results:
- Bathroom vs Area value:
    - Fail to reject the Null Hypothesis. This suggests there is more value in homes with above median bathrooms than homes with above median area.
    - Fail to reject the Null Hypothesis. This suggets there is more value in homes with below median bathrooms than homes with below median area.
- Bedroom vs Area value:
    - Reject the Null Hypothesis. This suggests there is more value in homes with above median area than homes with above median bedrooms.
    - Fail to reject the Null Hypothesis. This suggets there is more value in homes with below median bedrooms than homes with below median area.
- Bedroom vs Bathroom value:
    - Fail to reject the Null Hypothesis. This suggests there is more value in homes with above median bathrooms than homes with above median bedrooms.
    - Reject the Null Hypothesis. This suggests there is more value in homes with below median bedrooms than homes with below median bathrooms.

#### Summary:
This one actually had some unexpected results, such as where bathrooms rank on what is more valuable, and where on the lower side of pricing, bedrooms have the largest affect on value.

***

## <a name="model"></a>Modeling:
[[Back to top](#top)]

### Model Preparation:
Ran some baseline correlation tests to compare variables to the target tax_value. Found Area and Bathrooms still had the best correlation, though not an overly strong one. Went on to test for the best variables, finding that Beds and Bath, and Rooms and Area were also intersting combinations to add to the mix, and kept in Bedrooms due to it's affect on the lower end of the pricing.

### Baseline
    
- Baseline Results: 
    RMSE of Mean is lower so we'll use that : 243788.25256933476


- Selected features to input into models:
    - features = [area, bedrooms, bathrooms, beds_and_bath, rooms_and_area]

***

### Models and R<sup>2</sup> Values:
- Will run the following regression models:
    - OLS
    - Lasso Lars
    - GLM
    
Validation/Out-of-Sample is done on the Validate dataframe.
    
#### Model 1: Linear Regression (OLS)


- Model 1 results:

Training/In-Sample:  216856.18050469778 
Validation/Out-of-Sample:  220539.4747717098

### Model 2 : Lasso Lars Model


- Model 2 results:

Training/In-Sample:  216853.2152719072 
Validation/Out-of-Sample:  220542.06464866508

### Model 3 : Tweedie Regressor (GLM)

- Model 3 results:

Training/In-Sample:  243788.25256933476 
Validation/Out-of-Sample:  247065.52052137486


## Selecting the Best Model:

### Use Table below as a template for all Modeling results for easy comparison:

| Model | Validation/Out of Sample RMSE | R<sup>2</sup> Value |
| ---- | ----| ---- |
| Baseline | 5.943271e+10 | 0.0 |
| Linear Regression (OLS) | 2.205395e+05 | 0.203221 |  
| Tweedie Regressor (GLM) | 2.470655e+05 | 0.0 |  
| Lasso Lars | 2.205523e+05 | 0.203114 |  
| Quadratic Regression | 2.205430e+05 | 0.203182 |  

- {Lasso Lars} model performed the best


## Testing the Model

- Model Testing Results
RMSE for OLS Model using Lasso + Lars
Out-of-Sample Performance:  214818.064501288

***

## <a name="conclusion"></a>Conclusion:
[[Back to top](#top)]
There is a lot more to home price than the typical "How many Bedrooms?", "How many Bathrooms?", "How big is it?" that you'd normally hear someone ask you about a house.
There is so much more that could be done with the other variables that I'd love time to come back and delve further into how things like Lot size, Location, Pool, Building materials, etc have on a property.

***

###### ReadMe template by: <a href="https://github.com/mdalton87" target="_blank">Mathew Dalton</a>
</p>
