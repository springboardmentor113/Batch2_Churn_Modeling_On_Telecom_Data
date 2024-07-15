#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.ticker as mtick
from statsmodels.stats.outliers_influence import variance_inflation_factor


# # Reading and understanding the data

# In[2]:


# Reading the dataset
Churn_Data = pd.read_csv('Churn_ Data.csv')
Churn_Data.head()


# In[3]:


Churn_Data.describe()


# ### Q1)Convert data type of variables which are misclassified.

# In[4]:


Churn_Data.dtypes.value_counts()


# In[5]:


#checking the datatypes.
pd.set_option('display.max_rows', None)
Churn_Data.dtypes


# In[6]:


Churn_Data.info()


# ### Q2) Removing Duplicate records
# 

# In[7]:


print ("Number of the record befor removing duplicates:", len (Churn_Data))
Churn_Data.drop_duplicates(inplace=True)

print("Number of the record after removing duplicates:" , len (Churn_Data))


# In[8]:


Churn_Data.shape


# ### Q3) Removing unique value variables
# 

# In[9]:


num_columns_before = len (Churn_Data.columns)
for columns in Churn_Data.columns:
    if len(Churn_Data[columns].unique()) == 1:
        Churn_Data.drop(column, axis=1, inplace = True)
num_columns_after = len(Churn_Data.columns) 

print ("Number of columns before removing unique value variables:", num_columns_before)
print ("Number of columns after removing unique value variables:", num_columns_after)


# In[10]:


Churn_Data.shape


# ### Q4) Removing Zero variance variables
# 

# In[11]:


num_columns_before = len (Churn_Data.columns)
for column in Churn_Data.columns:
    if Churn_Data[column].var() == 0:
        Churn_Data.drop(column, axis=1, inplace = True)
num_columns_after = len(Churn_Data.columns) 

print ("Number of columns before removing zero variance variables:", num_columns_before)
print ("Number of columns after removing zero variance variables:", num_columns_after)


# In[12]:


Churn_Data.shape


# ## Q6) Outlier Treatment

# ### Using Boxplot: Q3+(1.5*IQR) & Q1-(1.5*IQR)

# In[13]:


def detect_outliers_iqr(data_column, multiplier= 1.5):
    Q1 = np.percentile(data_column, 25)
    Q3 = np.percentile(data_column, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (data_column < lower_bound) | (data_column > upper_bound)

multiplier = 2.5
outliers = {}
for column in Churn_Data.select_dtypes(include=np.number). columns:
    outliers[column] = detect_outliers_iqr(Churn_Data[column], multiplier=multiplier)
    
#Outliers Counts
outlier_counts = {col: outliers[col].sum() for col in outliers}
for col, count in outlier_counts.items():
    print (f"Column '{col}': {count} Outliers Detected ")
    
#Removing Outliers
def remove_outliers(df, multiplier= 1.5):
    numeric_columns = df.select_dtypes(include= ['float64','int64']).columns
    for column in numeric_columns:
        outlier_mask = detect_outliers_iqr(df[column], multiplier=multiplier)
        df = df[~outlier_mask]
    return df

#Make a copy  of the original dataframe before removing outliers
Churn_Data_Clean = Churn_Data.copy()
Churn_Data_Clean = remove_outliers(Churn_Data_Clean, multiplier=multiplier)

print ("Number of record after outlier treatment:", len (Churn_Data_Clean))


# In[14]:


for column in Churn_Data.select_dtypes(include=np.number):
    plt.figure(figsize= (6, 4))
    plt.boxplot(Churn_Data[column], vert= False)
    plt.title(f"Box Plot for {column}")
    plt.show()


# ### Standardization: +/- 3 Sigma approach
# 

# In[16]:


#Function to detect outlier using Z-score
def detect_outlier_zscore(data_column):
    threshold = 3
    mean = np.mean(data_column)
    std_dev = np.std(data_column)
    z_scores = np.abs((data_column - mean) / std_dev)
    return z_scores > threshold

#Detect outlier for each numeric columns using z-score approach
outlier  = {}
for column in Churn_Data.select_dtypes(include=np.number):
    outlier[column] = detect_outlier_zscore(Churn_Data[column])
    

#Remove outlier from the dataset
def remove_outlier_zscore(df, outliers):
    for column in df.select_dtypes(include= ['float64','int64']).columns:
        df = df.loc[~outlier[column]]
    return df

Churn_Data_Clean = remove_outlier_zscore (Churn_Data, outliers)

#Print number of record after outlier treatment

print ("Number of records after outlier treatment:", len (Churn_Data_Clean))


# ### Capping & Flooring
# 

# In[17]:


#Function to cap and floot outlier based on percntile
def cap_floor_outlier_percentile (df, lower_percentile= 0.05, upper_percentile = 0.95):
    lower_bound = df.quantile(lower_percentile)
    upper_bound = df.quantile(upper_percentile)
    
#Apply capping and flooring
    for column in df.select_dtypes(include= ['float64', 'int64']).columns:
        df[column] = df[column].apply(lambda x: upper_bound[column] if x > upper_bound[column] else x)
        df[column] = df[column].apply(lambda x: upper_bound[column] if x < upper_bound[column] else x)
        
    return df

#Apply the function to the Chunr Data
Churn_Data_Clean = cap_floor_outlier_percentile(Churn_Data)

#Print number of records after outlier treatment
print ("Number of records after outlier treatment:", len (Churn_Data_Clean))

    


# ## Q5) Missing Value Treatment
# 

# In[15]:


#Removing records that have missing values in more than 5% of their columns
#Only records that have at least 95% of thier columns (non-NA values) will be retained
num_records_before = len (Churn_Data_Clean)
Churn_Data_Clean.dropna(thresh=len(Churn_Data_Clean.columns)* 0.95, inplace=True)
num_records_after = len (Churn_Data_Clean)
print("Number of records before removing if NA's are less than 5%:", num_records_before)
print("Number of records after removing if NA's are less than 5%:", num_records_after)

#Removing records if NA's are 50% in any variable
num_columns_before = len (Churn_Data_Clean.columns)
Churn_Data_Clean.dropna(thresh=len(Churn_Data_Clean) * 0.5, axis=1, inplace=True)
num_columns_after = len (Churn_Data_Clean.columns)
print ("Number of columns before removing if NA's are 50%:", num_columns_before)
print ("Number of columns after removing if NA's are 50%:", num_columns_after)

#Imputing with Mean/Median
numeric_cols = Churn_Data_Clean.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = Churn_Data_Clean.select_dtypes(include=['object']).columns

for column in numeric_cols:
    Churn_Data_Clean[column].fillna(Churn_Data_Clean[column].median(), inplace=True)
    
for column in numeric_cols:
    Churn_Data_Clean[column].fillna(Churn_Data_Clean[column].mode()[0], inplace=True)


# ## Q7) Removing the highly correlated variables
# 

# In[21]:


correlation_matrix = Churn_Data_Clean.corr().abs()
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any (upper[column] > 0.9)]
Churn_Data_Clean.drop(to_drop, axis=1, inplace=True)
print("Number of columns after removing highly correleted variables:", len(Churn_Data_Clean.columns))


# ### Q8) Multicollinearity (VIF > 5)

# In[24]:


def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    vif_data["VIF"] = vif_data["VIF"].replace([np.inf, -np.inf], 9999)
    return vif_data

if 'dependent variable' in Churn_Data_Clean.columns:
    X = Churn_Data_Clean.drop(columns = ['dependent variable'])
else:
    X = Churn_Data_Clean.copy()
vif_df = calculate_vif(X)
print ("Number of columns after removing Multicollinearity:", len(Churn_Data_Clean.columns))


# ## Model Building 

# ### Decision Tree

# In[16]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[17]:


X = Churn_Data_Clean.drop(columns = ['target'],axis=1)
y = Churn_Data_Clean['target']

X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)


# In[25]:


initial_model = DecisionTreeClassifier(random_state = 42)
initial_model.fit(X_train,y_train)

initial_y_pred = initial_model.predict(X_test)

print ("initial Model Confusion Matrix:")
print (confusion_matrix(y_test,initial_y_pred ))

print ("\ninitial Model Classification Report:")
print (classification_report(y_test, initial_y_pred))

print ("\ninitial Model Accuracy Score:")
print (accuracy_score(y_test, initial_y_pred))
acc_prec = accuracy_score(y_test, initial_y_pred)* 100
print ("\nAccuracy Percentage:", round(acc_prec, 3), "%")


# ### Hyperparameter tuning for Decision tree Model

# In[26]:


#Defining the parameter grid
param_grid  = {
    'max_depth': [None, 10,20],
    'max_features': ['sqrt', 'log2', None]
}

#Create the dicision model
dt_model= DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

#Print the best parameters found by GridSearchCV

print ("\nBest parameters found by GridSearchCV:")
print (grid_search.best_params_)


best_model = grid_search.best_estimator_
tuned_y_pred = best_model.predict(X_test)

print ("\nTuned Model Confusion Matrix:")
print (confusion_matrix(y_test,tuned_y_pred ))

print ("\nTuned Model Classification Report:")
print (classification_report(y_test, tuned_y_pred))

acc_prec = accuracy_score(y_test, tuned_y_pred)* 100
print ("\nTuned Model Accuracy Percentage:", round(acc_prec, 3), "%")


# ### Random Forest

# In[20]:


rf_model = RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(X_train,y_train)
y_pred = rf_model.predict(X_test)

initial_y_pred=rf_model.predict(X_test)

print("Initial Model Confusion Matrix:")
print(confusion_matrix(y_test,initial_y_pred))

print ("\nInitial Model Classification Report:")
print (classification_report(y_test, initial_y_pred))

print("\nInitial Model Accuracy Score:")
print(accuracy_score(y_test,initial_y_pred))
acc_prec = accuracy_score(y_test, initial_y_pred)* 100
print ("\n Accuracy Percentage:", round(acc_prec, 3), "%")


# ### Hyperparameter tuning for Random Forest Model

# In[31]:


param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'max_features': ['sqrt', 'log2']
}

#Create the random Forest model

rf_model = RandomForestClassifier()

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print ("Best Hyperparameters:", best_params)

best_rf_model = RandomForestClassifier(**best_params)
best_rf_model.fit(X_train, y_train)

best_model = grid_search.best_estimator_
tuned_y_pred = best_model.predict(X_test)

#Evaluate the tuned model
print("\nTuned Model Confusion Matrix:")
print(confusion_matrix(y_test, tuned_y_pred))

print("\nTuned Model Classification Matrix:")
print(classification_report(y_test, tuned_y_pred))

acc_prec = accuracy_score(y_test, tuned_y_pred)* 100
print ("\nTuned Model Accuracy:", round(acc_prec, 3), "%")



# ### Logistic Regression

# In[34]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=5000, solver="liblinear")

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report= classification_report (y_test, y_pred)


acc_prec = accuracy_score(y_test, y_pred)*100
print("\nAccuracy Percentage:", round(acc_prec, 3), "%")
print ("Confusion Matrix:")
print (conf_matrix)
print ("Classification Report:")
print (class_report)



# ### Hyperparameter tuning forLogistic Regressiont Model

# In[36]:


param_grid = {
    'solver': ['liblinear','saga'],
    'penalty': ['l1','l2'],
    'C':[0.01,0.1,1,10,100],
    'max_iter':[1000,2000,5000]
}

logreg = LogisticRegression()

grid_search = GridSearchCV(logreg, param_grid, cv=5, verbose=1,n_jobs =-1)

grid_search.fit(X_train_scaled,y_train)

best_params = grid_search.best_params_

print(f'Best parameters: {best_params}')

best_estimator = grid_search.best_estimator_
y_pred_best = best_estimator.predict(X_test_scaled)

accuracy_best = accuracy_score(y_test,y_pred_best)
conf_matrix_best = confusion_matrix(y_test,y_pred_best)
class_report_best = classification_report(y_test,y_pred_best)

print(f'Accuracy of Best Model: {accuracy_best}')
print('Confusion Matrix of Best Model:')
print (conf_matrix_best)
print ("Classification Report of Best Model:")
print (class_report_best)

acc_prec_best = accuracy_best * 100
print("\nAccuracy Percentage of Best Model:", round (acc_prec_best, 3), "%")


# ### Plotting the ROC Curve

# In[1]:


# ROC Curve function

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[4]:


draw_roc(y_train['churn'], y_train['churn_prob'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




