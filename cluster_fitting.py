# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 18:12:56 2023

@author: Ayush Jha
@Assignment: Cluster and fitting
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import scipy.stats as sci
import scipy.optimize as sciop



# Function to read and transpose the file
def read_Data(fileName):
    """
    Read,Clean and Transpose.

    function read the data from the csv file,Clean the data to make
    it usable and transpose the data.transposed data further get cleaned to
    include header as needed
    """
    df_Data1 = pd.read_csv(fileName,index_col=False)
# Using trimData function to remove unwanted data
    df_Data = trimData(df_Data1)
    df_dataTransposed = df_Data.T
    return df_Data, df_dataTransposed

def cleandata(dataset):
    data = dataset.dropna(axis=1)
    data = data.reset_index()
    data = data.iloc[:,1:6]
    return data

def trimData(data):
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

def get_Gdp(Dataset):
    data= pd.read_csv('GDP.csv');
    data = trimData(data)
    data_year = data.iloc[:,34:]
    data_country = data.iloc[:,0]
    data = pd.concat([data_country,data_year],axis=1)
    data = data.reset_index()
    data = data.iloc[:,1:]
    data_transpose = data.T
    data = cleanDataTrans(data_transpose)
    data['Year']= data.index
    data['Year']=pd.to_numeric(data['Year'])
    return data

def curvefunction(x,a,b):
    y = a + np.exp(b*x)
    return y


# Reading the data by using read data function
df_data, df_data_trans = read_Data('Data.csv')
df_data = cleandata(df_data) 
df_data_trans = cleanDataTrans(df_data_trans)
Gdp_data = get_Gdp(df_data)
print(Gdp_data.dtypes)


#plot

plt.figure()
plt.plot(Gdp_data["Year"], Gdp_data["India"], label="India")
plt.plot(Gdp_data["Year"], Gdp_data["China"], label="China")
plt.plot(Gdp_data["Year"], Gdp_data["United States"], label="US")
plt.plot(Gdp_data["Year"], Gdp_data["United Kingdom"], label="UK")
plt.plot(Gdp_data["Year"], Gdp_data["Brazil"], label="Brazil")
plt.plot(Gdp_data["Year"], Gdp_data["Australia"], label="Australia")
plt.plot(Gdp_data["Year"], Gdp_data["South Africa"], label="South Africa")
plt.plot(Gdp_data["Year"], Gdp_data["Germany"], label="Germany")
plt.title("GDP changes of each country over past 30 year",fontsize=12)
plt.xticks(rotation = 90)
plt.legend()
plt.show()

# scatter
plt.scatter(Gdp_data["Year"], Gdp_data["China"])
plt.xticks(rotation =90)

param, pcovar = sciop.curve_fit(curvefunction, Gdp_data["Year"], Gdp_data["China"],maxfev=40000)
print(*param)
plt.plot(Gdp_data["Year"],curvefunction(Gdp_data["Year"],*param))
plt.plot(Gdp_data["Year"],curvefunction(Gdp_data["Year"]))


#heatmap

corr=df_data.describe().corr()
fig, ax = plt.subplots(figsize=(8, 8))
ax.matshow(corr, cmap='coolwarm')
# setting ticks to column names
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

 
plt.figure()
data = list(zip(df_data['GDP(PPP)'],df_data['Population']))
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
plt.scatter(df_data['GDP(PPP)'],df_data['Population'], c=kmeans.labels_)
plt.show()

plt.figure()
data = list(zip(df_data['GDP(PPP)'],df_data['CO2 Consumption']))
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
plt.scatter(df_data['GDP(PPP)'],df_data['CO2 Consumption'], c=kmeans.labels_)
plt.show()


