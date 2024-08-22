<H3>ENTER YOUR NAME : AADITHYAN R</H3>
<H3>ENTER YOUR REGISTER NO : 212222230001</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 22.8.24</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv('/content/Churn_Modelling.csv')
print(df.head())

X=df.iloc[:,:-1].values
print(X)

y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())

df.duplicated()

df.describe()

df = df.drop(['Surname', 'Geography','Gender'], axis=1)
df.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(X_train)
print(len(X_train))

print(X_test)
print(len(X_test))
```
## OUTPUT:
### DATASET:
![1](https://github.com/user-attachments/assets/63cfc27c-6c61-48ca-a78f-823629c3d564)

### X VALUES:
![2](https://github.com/user-attachments/assets/cb69b34e-382c-4865-ae99-cc56070ba46e)

### Y VALUES:
![3](https://github.com/user-attachments/assets/91285301-aed9-4455-a899-994d73ad0950)

### NULL VALUES:
![4](https://github.com/user-attachments/assets/741f94bf-c1c1-477f-91cb-5388f04878a8)

### DUPLICATED VALUES:
![5](https://github.com/user-attachments/assets/4b4c2a61-217d-410b-90d0-84f08ad5282f)

### DESCRIPTION:
![6](https://github.com/user-attachments/assets/15cabb08-6f19-4002-a8ee-0932040b6afd)

### NORMALIZED DATASET:
![7](https://github.com/user-attachments/assets/9020fe99-6e44-4871-ab06-4fb9279b43af)

### TRAINING DATASET:
![8](https://github.com/user-attachments/assets/46600a57-4529-4c57-a68e-1461af2f0cff)

### TESTING DATASET:
![9](https://github.com/user-attachments/assets/7353c8e4-d2af-4bd5-aa78-4d9ba81fb5e3)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


