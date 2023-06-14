#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 02:43:35 2023

@author: myyntiimac
"""

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv("/Users/myyntiimac/Desktop/emp_sal.csv")
df.head()
df.shape# this data contain 10rows and 3 columnn . thats very small data, so we are not going to split test train,because the data 
#size going to very small  so that model not apropriatly predict (as we wanna build model which predict salary)instead we use the whole data for training
# Check the data have null
df.isnull().any()# we find false that means , no null , so we are not performing EDA
#By see the data frame we found that position and level are corresponds each other , so we delete position attribute, , because its has no effct in independent
# And by business understanding of data, salary will be dependent and level will be independent.
#Lets delete position column , and define independent and depenndent by slicing the df
x = df.iloc[:, 1:2].values
#now we will create X matrix of feature and we will spcify the index 1 - LEVEL & however there is something
#then i have to mention 2 because in python upper bound of range is excluded
y = df.iloc[:, 2].values
#Dependent variable we will specify the index 2
#Lets check how the data behave , linera, or polynomian by scatterplot
plt.scatter(x, y, color = 'green')# we see the data is not lenear, is pölynomial, so we build plonomial regression 
# we also build LR model, why? we will compare the result of both model
# anouther point , by obserbing data, we see that salary range is higher than position range, so we need to do feature scaling for uniforming the range.
# but in this cas we dont do feature scaling, because we convert x and y to polynomial metrix by polynomial feature
#Lets make polynomial feature and make model
#In this section we build the linear regression model & also we gonna build the polynomial regression model to fit the dataset

# Fitting Linear Regression to the dataset

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
#To compare both linear model created lin_reg & for polynomial we will mention as lin_reg2
#so create a object of lin_reg and called the class LinearRegression
lin_reg.fit(x, y)
#fit the lin_reg object to X & y. now our simple linear regression is ready 


#Fitting the polynomial regression to the dataset and build polynomial model
#to create this model we will import a PolynomialFeature
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures() # we mentoine 2 degree , is by default, but we can change it
#create an object called poly_reg & we will assigned the degree 

#now convert the X to X_matrix by plynomial feature
X_poly = poly_reg.fit_transform(x)
# check X_poly with defferent degrre instead of default 2
poly_reg1= PolynomialFeatures(4)
X_poly1 = poly_reg1.fit_transform(x)
# now fit the data with ply_reg and build Plynomial model
poly_reg.fit(X_poly, y)
#now we have to fit the poly_reg fit instad of x we have to fit with X_poly, y

# Lets build another LINear reg model with X_ply feature
from sklearn.linear_model import LinearRegression
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

## Now we have two lenear model one with actual x ankouther one with x_poly( converted ply matrix), and ply_reg model
# now we will see the actual and predicted point by plot visualization
#lets starts the plotting by true observation 
plt.scatter(x, y, color = 'blue')
#we are going to plot for actual value of X & y
plt.plot(x, lin_reg.predict(X), color = 'green')
#now plot for the prediction line where x coordinate are predictin points & for y-cordinates predicted value which is lin_reg.predict (x)
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# in here we see , our actual point in blue point not fitted with our predicted Linear regression line because our data in ploynomial nature
# so, this is not a good prediction, only two points capture by predicted line.Most actual points is ignore
#so we need better model where actual and predicted line match
#next step is we will build the non-linear model sothat its fits with our polynomial data in nature(checking data neture)
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#you can easily say that distinguish b/w linear model & non-linear model
#linear model is straigt line but non-linear is not a straight line
#But its still not good fit, how can we make best fited with our plynomial data and make good model which predicted well
# we can do it by changing the degree of polynomial feature
#Lets build the  anouther linear model with polynomial degree feature 4(previously did)
# now fit the data with ply_reg and build Plynomial model
poly_reg1.fit(X_poly1, y)
#now we have to fit the poly_reg fit instad of x we have to fit with X_poly, y

# Lets build another LINear reg model with X_poly1 feature
from sklearn.linear_model import LinearRegression
lin_reg3 = LinearRegression()
lin_reg3.fit(X_poly1, y)

# then chck the improvement
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg3.predict(poly_reg1.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#we  find the with plynomial feature degree , the predicted and actual line is best fitted , that means 
#predicted model capture all the values which naturaly polynomial nature(previuosly checked with scatter.)
# Now we will predict the unknown level and our new interview candidate who claim more than 150,000/annual in level-7
# First , predict the claim with linear regression
lin_reg.predict([[6.5]])
#secondly, predict the calim with optimized polynomial model
lin_reg3.predict(poly_reg1.fit_transform([[6.5]]))
# Lets see how lour optimized polynomial predict totaly unkown for them (out of range level, not trained this range)
lin_reg3.predict(poly_reg1.fit_transform([[12]]))

######Note : we can more optimized this model by changinng the degree like 5,6 etc




