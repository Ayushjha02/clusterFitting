# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 18:12:56 2023.

@author: Ayush Jha
@Assignment: Cluster and fitting
"""

import pandas as pd
import matplotlib.pyplot as plt
import itertools as iter
import numpy as np
from sklearn.cluster import KMeans
import scipy.optimize as opt


# Function to read and transpose the file
def read_Data(fileName):
    """
    Read,Clean and Transpose.

    function read the data from the csv file,Clean the data to make
    it usable and transpose the data.transposed data further get cleaned to
    include header as needed
    """
    df_Data1 = pd.read_csv(fileName, index_col=False)
# Using trimData function to remove unwanted data
    df_Data = trimData(df_Data1)
    df_dataTransposed = df_Data.T
    return df_Data, df_dataTransposed


# Function to clean dataset
def cleandata(dataset):
    """Remove the not available rows and reset the index."""
    data = dataset.dropna(axis=1)
    data = data.reset_index()
    data = data.iloc[:, 1:6]
    return data


# Function to trim data
def trimData(data):
    """Remove all the countries except the one mentioned in the list below."""
    Countries = ['China', 'India', 'United States', 'United Kingdom',
                 'Germany',
                 'Brazil', 'Australia', 'South Africa']
    trimmed_data = data[data['Country Name'].isin(Countries)]
    return trimmed_data


# Function to  clean the transposed data
def cleanDataTrans(dataframe):
    """
    Clean the transposed data.

    function clean the data by removing unwanted header
    and change it to use country name
    """
    dataframe.rename(columns=dataframe.loc["Country Name"], inplace=True)
    dataframe = dataframe.drop(["Country Name"], axis=0)
    dataframe = dataframe.dropna()
    return dataframe


# Function to read and transpose the file
def read_data2(dataset,):
    """
    Read,Clean and Transpose.

    function read the data from the csv file,Clean the data to make
    it usable and transpose the data.transposed data further get cleaned to
    include header as needed
    """
    data = pd.read_csv(dataset)
    data = trimData(data)
    data_year = data.iloc[:, 34:]
    data_country = data.iloc[:, 0]
    data = pd.concat([data_country, data_year], axis=1)
    data = data.reset_index()
    data = data.iloc[:, 1:]
    data_transpose = data.T
    data = cleanDataTrans(data_transpose)
    data['Year'] = data.index
    data['Year'] = pd.to_numeric(data['Year'])
    return data


# Function to find Error range
def err_ranges(x, func, param, sigma):
    """
    Calculate the upper and lower limits for the function.

    parameters and sigmas for single value or array x.
    Functions values are calculated for
    all combinations of +/- sigma and the minimum and maximum is determined.

    Can be used for all number of parameters and sigmas >=1.
    """
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper


# Function to fit curve on data
def curvefunction(x, a, b, c):
    """
    Find the data for curve fit.

    Function uses polynomials with degree 2 to get suitable
    curve fit and predict the data.
    """
    y = a + b*x + c*x**2
    return y
# Function for Error


def Error(data):
    """
    Pass CO2 data into a exp function.

    find error for sigma parameter used in curve fit.
    """
    Error = np.log(data)
    return Error


def norm(array):
    """
    Return array normalised to [0,1].

    Array can be a numpy array
    or a column of a dataframe
    """
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled


# Reading the data by using different CSV file
df_data, df_data_trans = read_Data('Data.csv')
Gdp_data = read_data2('GDP.csv')
co2_data = read_data2('co2.csv')
ele_data = read_data2('Electricity.csv')

# Cleaning the data
df_data_trans = cleanDataTrans(df_data_trans)
df_data = cleandata(df_data)


# Plot for displaying GDP
plt.figure(figsize=(10, 10))
plt.plot(Gdp_data["Year"], Gdp_data["India"], label="India")
plt.plot(Gdp_data["Year"], Gdp_data["China"], label="China")
plt.plot(Gdp_data["Year"], Gdp_data["United States"], label="US")
plt.plot(Gdp_data["Year"], Gdp_data["United Kingdom"], label="UK")
plt.plot(Gdp_data["Year"], Gdp_data["Brazil"], label="Brazil")
plt.plot(Gdp_data["Year"], Gdp_data["Australia"], label="Australia")
plt.plot(Gdp_data["Year"], Gdp_data["South Africa"], label="South Africa")
plt.plot(Gdp_data["Year"], Gdp_data["Germany"], label="Germany")
plt.title("GDP changes of each country over past 30 year", fontsize=10)
plt.xlabel("Year")
plt.ylabel("GDP")
plt.xticks(rotation=90)
plt.legend()
plt.show()


# Bar graph to display GDP/Capita
plt.figure(figsize=(10, 10))
countries = np.array(Gdp_data.columns[0:-1])
df_data["GDP/Capita"] = df_data["GDP(PPP)"]/df_data["Population"]
Capita = np.array(df_data["GDP/Capita"])
plt.ylabel("GDP/Capita")
plt.xlabel("Countries")
plt.title('GDP/Capita of all the countries')
plt.bar(countries, Capita, width=0.5)
plt.show()

# Plot to display CO2 emmision
plt.figure(figsize=(10, 10))
plt.plot(co2_data["Year"], co2_data["India"], label="India")
plt.plot(co2_data["Year"], co2_data["China"], label="China")
plt.plot(co2_data["Year"], co2_data["United States"], label="US")
plt.plot(co2_data["Year"], co2_data["United Kingdom"], label="UK")
plt.plot(co2_data["Year"], co2_data["Brazil"], label="Brazil")
plt.plot(co2_data["Year"], co2_data["Australia"], label="Australia")
plt.plot(co2_data["Year"], co2_data["South Africa"], label="South Africa")
plt.plot(co2_data["Year"], co2_data["Germany"], label="Germany")
plt.title("CO2 emmision of each country over past 30 year", fontsize=10)
plt.xlabel("Year")
plt.ylabel("CO2 emmision(Kt)")
plt.xticks(rotation=90)
plt.legend(loc=2, prop={'size': 10})
plt.show()


# Finding Curve_fit and Error for sigma
error = Error(co2_data["Year"])
param, pcovar = opt.curve_fit(
    curvefunction, co2_data["Year"], co2_data["India"], sigma=(error))


# predicted data for year 2019-2030
year = np.arange(2019, 2030)
predicted_data = curvefunction(year, *param)

# lower,upper error range
lower, upper = err_ranges(co2_data["Year"], curvefunction, param, error)

# Scatter plot using all the above data
plt.figure()
plt.scatter(co2_data["Year"], co2_data["India"])
plt.xticks(rotation=90)
plt.plot(co2_data["Year"], curvefunction(
    co2_data["Year"], *param), label="fit")
plt.plot(year, predicted_data, label="predicted line")
plt.xlabel("Year")
plt.ylabel("Co2 Consumption")
plt.title("Co2 emmission of india from 1990-2030")
plt.legend()
plt.show()

# Plot to display Electricity using renewable source
plt.figure(figsize=(10, 10))
plt.plot(ele_data["Year"], ele_data["India"], label="India")
plt.plot(ele_data["Year"], ele_data["China"], label="China")
plt.plot(ele_data["Year"], ele_data["United States"], label="US")
plt.plot(ele_data["Year"], ele_data["United Kingdom"], label="UK")
plt.plot(ele_data["Year"], ele_data["Brazil"], label="Brazil")
plt.plot(ele_data["Year"], ele_data["Australia"], label="Australia")
plt.plot(ele_data["Year"], ele_data["South Africa"], label="South Africa")
plt.plot(ele_data["Year"], ele_data["Germany"], label="Germany")
plt.title("Electric production from renewable source", fontsize=12)
plt.xlabel("Year")
plt.ylabel("ELectricity production(Kwh)")
plt.xticks(rotation=90)
plt.legend(loc=2, prop={'size': 8})
plt.show()


# Heatmap to show the correlation between different parameters
corr = df_data.iloc[:, :-1].corr()
fig, ax = plt.subplots(figsize=(10, 10))
ax.matshow(corr, cmap='coolwarm')
# setting ticks to column names
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()


# Using Normalization
GDP = norm(np.array(df_data['GDP(PPP)']))
CO2 = norm(np.array(df_data['CO2 Consumption']))
# Showing data using K-means cluster
plt.figure(figsize=(8, 8))
data = list(zip(GDP, CO2))
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
plt.scatter(GDP, CO2, c=kmeans.labels_)
plt.title("GDP vs CO2 Emmision ")
plt.xlabel("GDP(PPP)")
plt.ylabel("CO2 Emmision")
plt.show()
