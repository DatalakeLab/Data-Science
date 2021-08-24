# coding: utf-8
 
# Here are what the columns represent:
# * credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# * purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# * int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# * installment: The monthly installments owed by the borrower if the loan is funded.
# * log.annual.inc: The natural log of the self-reported annual income of the borrower.
# * dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# * fico: The FICO credit score of the borrower.
# * days.with.cr.line: The number of days the borrower has had a credit line.
# * revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# * revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# * inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# * delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# * pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

# # Import Libraries
# 
# **Import the usual libraries for pandas and plotting. You can import sklearn later on.**

import pandas as pd
import numpy as np
import psutil
import resource
import memory_profiler
from memory_profiler import profile
import time


def uso(mensagem):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print ("**********************************************")
    print (mensagem)
    #print ("CPU Usage: " + str(psutil.cpu_times()))
    print ("***CPU Percent:None: " + str(psutil.cpu_percent(interval=None, percpu=True)))
    #print ("CPU Percent:1: " + str(psutil.cpu_percent(interval=1, percpu=True)))
    #print ("CPU Percent:0.1: " + str(psutil.cpu_percent(interval=0.1, percpu=True)))
    mem_usage = memory_profiler.memory_usage()[0]
    print ("***Memory Usage: " + str(mem_usage))
    #for name, desc in [
    # ('ru_utime', 'User time'),
    # ('ru_stime', 'System time'),
    # ('ru_maxrss', 'Max. Resident Set Size'),
    # ('ru_ixrss', 'Shared Memory Size'),
    # ('ru_idrss', 'Unshared Memory Size'),
    # ('ru_isrss', 'Stack Size'),
    # ('ru_inblock', 'Block inputs'),
    # ('ru_oublock', 'Block outputs'),
    # ]:
    # print '%-25s (%-10s) = %s' % (desc, name, getattr(usage, name))


# ## Get the Data
# 
# ** Use pandas to read loan_data.csv as a dataframe called loans.**

start = time.time()
uso ("Reading Data")
loans = pd.read_csv('loan_data.csv')


# ** Check out the info(), head(), and describe() methods on loans.**


uso ("Info")
loans.info()

uso ("Description")
loans.describe()

uso ("Head")
loans.head()


# # Exploratory Data Analysis
# 
# Let's do some data visualization! We'll use seaborn and pandas built-in plotting capabilities, but feel free to use whatever library you want. Don't worry about the colors matching, just worry about getting the main idea of the plot.
# 
# ** Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.**
# 
# *Note: This is pretty tricky, feel free to reference the solutions. You'll probably need one line of code for each histogram, I also recommend just using pandas built in .hist()*


# ** Let's see the trend between FICO score and interest rate. Recreate the following jointplot.**


# ** Create the following lmplots to see if the trend differed between not.fully.paid and credit.policy. Check the documentation for lmplot() if you can't figure out how to separate it into columns.**


# # Setting up the Data
# 
# Let's get ready to set up our data for our Random Forest Classification Model!
# 
# **Check loans.info() again.**

uso ("Info Again")
loans.info()


# ## Categorical Features
# 
# Notice that the **purpose** column as categorical
# 
# That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies.
# 
# Let's show you a way of dealing with these columns that can be expanded to multiple categorical features if necessary.
# 
# **Create a list of 1 element containing the string 'purpose'. Call this list cat_feats.**

cat_feats = ['purpose']


# **Now use pd.get_dummies(loans,columns=cat_feats,drop_first=True) to create a fixed larger dataframe that has new feature columns with dummy variables. Set this dataframe as final_data.**


final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)

uso ("Final Data")
final_data.info()


# ## Train Test Split
# 
# Now its time to split our data into a training set and a testing set!
# 
# ** Use sklearn to split your data into a training set and a testing set as we've done in the past.**


from sklearn.model_selection import train_test_split


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# ## Training a Decision Tree Model
# 
# Let's start by training a single decision tree first!
# 
# ** Import DecisionTreeClassifier**


from sklearn.tree import DecisionTreeClassifier


# **Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.**


dtree = DecisionTreeClassifier()


dtree.fit(X_train,y_train)


# ## Predictions and Evaluation of Decision Tree
# **Create predictions from the test set and create a classification report and a confusion matrix.**


predictions = dtree.predict(X_test)


from sklearn.metrics import classification_report,confusion_matrix

uso ("Classification Report")
print(classification_report(y_test,predictions))

uso ("Confusion Matrix")
print(confusion_matrix(y_test,predictions))


# ## Training the Random Forest model
# 
# Now its time to train our model!
# 
# **Create an instance of the RandomForestClassifier class and fit it to our training data from the previous step.**

from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(n_estimators=600)


rfc.fit(X_train,y_train)


# ## Predictions and Evaluation
# 
# Let's predict off the y_test values and evaluate our model.
# 
# ** Predict the class of not.fully.paid for the X_test data.**

predictions = rfc.predict(X_test)


# **Now create a classification report from the results. Do you get anything strange or some sort of warning?**

from sklearn.metrics import classification_report,confusion_matrix

uso ("Classification Report")
print(classification_report(y_test,predictions))

uso ("Confusion Matrix")
print(confusion_matrix(y_test,predictions))

stop = time.time()

print ("Total Time: " + str((stop-start)*1000))
