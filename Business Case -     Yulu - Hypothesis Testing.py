#!/usr/bin/env python
# coding: utf-8

# In[89]:


# Importing libraries - 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats
from scipy.stats import kstest
import statsmodels.api as sm
import warnings 
warnings.filterwarnings("ignore")
import copy


# ## Business Case: Yulu - Hypothesis Testing 

# ### About Yulu 
# 
# 1. Yulu is India's leading micro-mobility service provider, which offers unqiue vechile for the daily commute. Starting off as a mission to eliminate traffic congestion in India, Yulu provides the safest commute solution through a user  friendly mobile app to enable shared, solo and sustainable commuting. 
# 
# 2. Yulu zones are located at all the appropriate locations (including metro stations, bus stands, office spaces, residential areas, corporate offices etc) to make those first and last miles smooth, affordable, and convenient.
# 
# 3. Yulu has recently suffered considerable dips in its revenues. The have contracted a consulting company to understand the factors on which the demand for these shared electric cycles depends. Specifically, they want to understand the factors affecting the demand for these shared electric cycles in the Indian market. 

# ### Problem Statement - 
# 
# The company wants to know -
# 1. Which variable are siginificant in predicting the demand for shared electric cycles in the Indian market?
# 2. How well those variables describe the electric cycle demands?

# ### Analyzing Basic Metrics -

# In[3]:


data = pd.read_csv("bike_sharing.txt")


# In[4]:


data


# In[5]:


data.shape


# #### Dataset contains 10886 rows and 12 columns. 

# In[6]:


data.size


# In[7]:


data.info()


# 1. The columns season, holiday, workingday, weather, humidity, casual, registered, count are of integer datatype, and the column 
# datetime is object type and rest are of float.
# 2. There are no null values. 
# 

# In[ ]:





# In[8]:


data.isnull().sum()


# In[9]:


data.nunique()


# In[52]:


## Converting the datatype of datetime column from object to datetime -  

data["datetime"] = pd.to_datetime(data["datetime"])
cat_cols = ["season", "holiday", "workingday", "weather"]
for col in cat_cols:
    data[col] = data[col].astype("object")



# In[55]:


num_cols = ['temp', 'atemp', 'humidity', 'windspeed', 'casual','registered','count']
num_cols


# In[11]:


data.describe()


# ####  Time period of data when it is given - 

# In[12]:


data["datetime"].min()


# In[13]:


data["datetime"].max()


# In[14]:


data["datetime"].max() - data["datetime"].min()


# In[15]:


data["day"] = data["datetime"].dt.day_name()


# In[16]:


data.describe()


# These statistics provide insights into the central tendency, spread, and range of the numerical features in the dataset.
# 

# In[17]:


## Season: season (1: spring, 2: summer, 3: fall, 4: winter)
data["season"].replace({ 1 : "spring", 2 : "summer" , 3 : "fall", 4 : "winter"}, inplace = True)


# In[18]:


## Cheking the season contribution - 

np.round(data["season"].value_counts(normalize = True) * 100, 2)


# In[19]:


## Checking the whether day is a holiday or not in percent - 

np.round(data["holiday"].value_counts(normalize = True) * 100, 2)


# In[20]:


## Cheking its working day or holiday -   

np.round(data["workingday"].value_counts(normalize = True) * 100, 2)


# #### Workingday - 
# 
# 1 - neither weekend nor holiday    
# 
# 0 - it's holiday
# 

# In[21]:


np.round(data["weather"].value_counts(normalize = True) * 100, 2)


# #### Weather:
# 1: Clear, Few clouds, partly cloudy, partly cloudy
# 
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

# ### Distribution of seasons - 
# 
# 

# In[22]:


plt.figure(figsize = (8,6))

season = np.round(data["season"].value_counts(normalize = True) * 100, 2).to_frame()

plt.pie(x = season["season"], labels = season.index, autopct = '%.2f%%', explode = [0.01, 0.01, 0.01, 0.01])

plt.title("Distribution of Seasons", fontdict = {'fontsize' : 14, 'fontweight' : 400})

plt.show()


# ### Distribution of Holiday - 

# In[23]:


plt.figure(figsize = (8,6))

holiday = np.round(data["holiday"].value_counts(normalize = True) * 100, 2).to_frame()

plt.pie(x = holiday["holiday"], 
        explode = [0, 0.1], 
        labels = ["Non-Holiday", "Holiday"],
        autopct = "%.2f%%")

plt.title("Distribution of Holiday", fontdict = {'fontsize' : 14,
                                                 'fontweight' : 400})

plt.show()


# ### Distribution of Workingday - 

# In[24]:


plt.figure(figsize = (8,6))

workingday = np.round(data["workingday"].value_counts(normalize = True) * 100, 2).to_frame()

plt.pie(x = workingday["workingday"],
        explode = [0, 0.01],
        labels = ["Working Day", "Non-Working Day"],
        autopct = "%.2f%%")

plt.title("Distribution of Working-Day", fontdict = {'fontsize': 14,
                                                    'fontweight' : 400})

plt.show()


# ### Distribution of Weather -

# In[25]:


plt.figure(figsize = (8,6))

weather = np.round(data["weather"].value_counts(normalize = True) * 100, 2).to_frame()

plt.pie(x = weather["weather"],
        explode = [0.01, 0.01, 0.01, 0.01],
        labels = weather.index,
        autopct = "%.2f%%")

plt.title("Distribution of Weather", fontdict = {'fontsize' : 14,
                                                 'fontweight' : 400})

plt.show()


# ## Univariate Analysis - 

# In[26]:


# Distribution of season - Count Plot 

sns.countplot(data = data, x = "season")

plt.title("Distribution of Season")

plt.show()


# In[28]:


# Distribution of Workingday- 

sns.countplot(data = data, x = "workingday")
plt.title("Distribution of Working-Day")
plt.show()


# In[27]:


# Distribution of Holiday - 

sns.countplot(data = data, x = "holiday")
plt.title("Distribution of Holiday")
plt.show()


# In[100]:


# Distribution of Weather - 

sns.countplot(data = data, x= "weather")
plt.title("Distribution of Weather")
plt.show()


# In[30]:


#  Distribution of Temperature values in the dataset -  Histplot

sns.histplot(data = data, x = "temp", kde = True, bins = 40)
plt.title("Distribution of temperature")
plt.show()


# In[31]:


temp_mean = np.round(data["temp"].mean(),2)
temp_std = np.round(data["temp"].std(),2)

temp_mean, temp_std


# The mean and the standard deviation of the temp column is 20.23 and 7.79 degree celcius respectivley.

# In[32]:


# Cumulative distribution of temperature values - Histplot 

sns.histplot(data = data, x = "temp", kde = True, cumulative = True, stat = "percent")
plt.grid(axis = "y", linestyle = "--")
plt.yticks(np.arange(0, 101, 10))
plt.show()


# In[33]:


# Distribution of  feeling temperature in the dataset - Histplot

sns.histplot(data = data, x = "atemp", kde = True, bins = 40)
plt.show()
  


# In[34]:


atemp_mean = np.round(data["atemp"].mean(), 2)
atemp_std = np.round(data["atemp"].std(),2)

atemp_mean, atemp_std


# The mean and the standard deviation of the atemp column is 23.66 and 8.47 degree celcius respectively.

# In[35]:


# Distribution of humidity values in the dataset - Histogram plot 

sns.histplot(data = data, x = "humidity", kde = True, bins = 40)
plt.show()


# In[36]:


humidity_mean = np.round(data["humidity"].mean(),2)
humidity_std = np.round(data["humidity"].std(),2)

humidity_mean, humidity_std


# The mean and the standard deviation of the humidity column is 61.89 and 19.25 respectively.

# In[37]:


# Histogram plot for the registered feature, showing the distribution of ragistered users values in the dataset - 

sns.histplot(data = data, x = "registered", kde = True, bins = 40)
plt.show()


# ## Outliers Detection - 

# In[38]:


# Outliers Detection

columns = ['temp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
count = 1

plt.figure(figsize = (15, 20))
for i in columns:
    plt.subplot(3, 2, count)
    plt.title(f"Detecting outliers in '{i}' column")
    sns.boxplot(data = data, x = data[i], showmeans = True, fliersize = 2)
    plt.plot()
    count += 1


# ## Bivariate Analysis - 

# In[39]:


# Distribution of hourly count of total rental bikes across all seasons - Box plot 

plt.figure(figsize = (15,6))

sns.boxplot(data = data, x ="season", y = "count", hue = "workingday", showmeans = True)
plt.grid(axis = "y", linestyle = "--")

plt.title("Distribution of rental bikes hourly across all seasons",
          fontdict = {"size" : 14,
                      "fontweight" : 400})

plt.plot()


# The hourly count of total rental bikes is higher in the fall season, followed by the summer and winter season. It is generally low in the spring season.

# In[46]:


# Distribution of hourly count of total rental bikes across all weathers - Box Plot

plt.figure(figsize = (15, 6))

sns.boxplot(data = data, x = "weather", y = "count", hue = "workingday", showmeans = True)
plt.grid(axis = "y", linestyle = "--")

plt.title("Distribution of hourly count of total rental bikes across all weathers",
          fontdict = {"size" : 14,
                      "fontweight" : 400})

plt.plot()


# 1. The hourly count of total rental bikes is higher in the clear and cloudy weather, followed by the misty weather and rainy weather.
# 2. There are very less data for extreme weather conditions

# In[57]:


fig, axis = plt.subplots(nrows=2, ncols=3, figsize=(16, 12))
index = 0
for row in range(2):
    for col in range(3):
        sns.scatterplot(data=data, x=num_cols[index], y='count', ax=axis[row,col])
        index += 1
plt.show()



# 1. Whenever the humidity is less than 20, number of bikes rented is very very low.
# 2. Whenever the temperature is less than 10, number of bikes rented is less.
# 3. Whenever the windspeed is greater than 35, number of bikes rented is less.

# In[58]:


data.corr()["count"]


# In[59]:


sns.heatmap(data.corr(), annot = True)
plt.show()


# ### Effect of Working day on the number of electric cycles rented
# 
# 

# In[43]:


data.groupby( by = "workingday")["count"].describe()


# In[44]:


sns.boxplot(data = data, x = "workingday", y = "count" )
plt.plot()


# ### Hypothesis Testing - 1 

# Null Hypothesis (H0) :  Weather is independent of the season.
# 
# Alternative Hypothesis (Ha) : Weather is not independent of the season.
# 
# Significance Level (alpha) :
# alpha = 0.05
# 
# We will use chi-square test to test hypothesis defined above

# In[63]:


data_table = pd.crosstab(data["season"], data["weather"])
print("Observed values: ")
data_table


# In[67]:


val = stats.chi2_contingency(data_table)
expected_values = val[3]
expected_values


# In[69]:


nrows, ncols = 4, 4
   
dof = (nrows-1)*(ncols-1)
print("degrees of freedom: ", dof)

alpha = 0.05

chi_sqr = sum([(o-e)**2/e for o, e in zip(data_table.values, expected_values)])

chi_sqr_statistic = chi_sqr[0] + chi_sqr[1]
print("chi-square test statistic: ", chi_sqr_statistic)

critical_val = stats.chi2.ppf(q=1-alpha, df=dof)
print(f"critical value: {critical_val}")

p_val = 1-stats.chi2.cdf(x=chi_sqr_statistic, df=dof)
print(f"p-value: {p_val}")

if p_val <= alpha:
   print("\nSince p-value is less than the alpha 0.05, We reject the Null Hypothesis i.e. Weather is dependent on the season.")
else:
   print("Since p-value is greater than the alpha 0.05, We do not reject the Null Hypothesis i.e. Weather is independent of the season")


# ### Hypothesis Testing - 2

# Null Hypothesis (H0) : Working day has no effect on the number of cycles being rented.
#     
# Alternative Hypothesis (Ha) : Working day has effect on the number of cycles being rented.
#     
# Significance Level (alpha) :
# alpha = 0.05
# 
# We will use the 2-Sample T-Test to test the hypothesis defined above.

# In[70]:


group1 = data[data["workingday"] == 0]["count"].values
group2 = data[data["workingday"] == 1]["count"].values

np.var(group1), np.var(group2)


# Before conducting the two-sample T- Test we need to find if the given data groups have the same variance. If the ratio of the larger data groups to the small data group is less than 4:1 then we can consider that the given data groups have the equal variance.
# 
# Here, the ratio is 34040.70 / 30171.35 which less than 4:1

# In[71]:


stats.ttest_ind(a = group1, b = group2, equal_var = True)


# Since the p-value is greater than 0.05 so we can not reject the Null Hypothesis.
# We don't have the sufficient evidence to say that working day has effect on the number of cycles being rented.
# 
# 

# ### Hypothesis Testing - 3

# Null Hypothesis (H0) : Number of cycles rented is similiar in different weather and season.
# 
# Alternative Hypothesis (Ha) : Number of cycles rented is not similar in different weather and season.
#     
# Significance Level (alpha) :
# alpha = 0.05
# 
# Here, we will use the ANOVA to test the hypothesis defined above.
# 

# In[97]:


group1 = data[data["weather"] == 1]["count"].values
group2 = data[data["weather"] == 2]["count"].values
group3 = data[data["weather"] == 3]["count"].values
group4 = data[data["weather"] == 4]["count"].values

group5 = data[data["season"] == 1]["count"].values
group6 = data[data["season"] == 2]["count"].values
group7 = data[data["season"] == 3]["count"].values
group8 = data[data["season"] == 4]["count"].values


# In[98]:


# One - way anova - 

stats.f_oneway(group1, group2, group3, group4, group5, group6, group7, group8)


# In[ ]:





# ## Insights - 

# 1. In summer and fall seasons more bikes are rented as compared to other seasons.
# 2. It is also clear from the workingday also that whenever day is holiday or weekend, slightly more bikes are rented.
# 3. There is statistically significant dependency of weather and season based on the hourly total number of bikes rented.
# 4. The hourly total number of rental bikes is statistically different for different weathers.
# 5. Whenever its a holiday more bikes are rented.
# 6. Whenever there is rain, thunderstorm, snow or fog, there were less bikes were rented.
# 7. Whenever the humidity is less than 20, number of bikes rented is very very low.
# 8. Whenever the tempereature is less than 10, number of bikes rented is less.
# 9. Whenever the windspeed is greater than 35, number of bikes rented is less.
# 

# ## Recommendations - 

# 1. In summer and fall seasons the company should have more bikes in stock to be rented, because the demand in these seasons is higher as compared to other seasons.
# 2. Offer seasonal discounts or special packages to attract more customers during the spring and winter seasons to attract more customer during these periods.
# 3. Encourage customers to provide feedback and reviews on their biking experience. Collecting feedback can help identify areas for improvement, understand customer preferences, and tailor the services to better meet customer expectations.
# 4. Leverage social media platforms to promote the electric bike rental services. Share captivating visuals of biking experiences in different weather conditions, highlight customer testimonials, and engage with potential customers through interactive posts and contests. Utilize targeted advertising campaigns to reach specific customer segments and drive more bookings.
# 5.  Given that around 81% of users are registered, and the remaining 19% are casual, Yulu can tailor its marketing and communication strategies accordingly.
# 6.  Provide loyalty programs, exclusive offers, or personalized recommendations for registered users to encourage repeat business. For casual users, focus on providing a seamless rental experience and promoting the benefits of bike rentals for occasional use.
# 
# 7. With significance level of 0.05, workingday has no effect on the number of bikes being rented.
# 
# 8. In very low humid days, company should have less bikes in the stock to be rented.
# 9. Whenever temperatrue is less than 10 or in very cold days, company should have less bikes.
# 10. Anaylyze the demand patterns during different months and adjust the inventory accordingly. During months with lower rental counts such as January, February, and March, Yulu can optimize its inventory levels to avoid excess bikes.
#  
