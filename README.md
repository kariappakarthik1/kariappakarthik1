
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


seed = 7
print("Libraries updated")


# Set some standard parameters upfront
pd.options.display.float_format = '{:.2f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
print('pandas version ', pd.__version__)


# Load data set containing all the data from csv
df = pd.read_csv('Loan_Predict_Dataset.csv')
# Describe the data, Shape and how many rows and columns
print('Number of columns in the dataframe: %i' % (df.shape[1]))
print('Number of rows in the dataframe: %i\n' % (df.shape[0]))
print(list(df.columns))
print(df['Loan_Status'].value_counts(), '\n')
print( df.head(5), '\n' )
print( df.describe(), '\n' )
print(df.isnull().sum(), '\n')


# pandas DataFrame: replace nan values with average of columns
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())
print(df.isnull().sum(), '\n')
print('Number of columns in the dataframe: %i' % (df.shape[1]))
print('Number of rows in the dataframe: %i\n' % (df.shape[0]))


df.dropna(inplace=True)
print(df.isnull().sum(), '\n')
print('Number of columns in the dataframe: %i' % (df.shape[1]))
print('Number of rows in the dataframe: %i\n' % (df.shape[0]))

print("Preprocessing Stage, Visualization Level 1 Done")


sns.countplot(df['Gender'],hue=df['Loan_Status'],palette='Set1')
print(pd.crosstab(df['Gender'],df['Loan_Status']))
plt.savefig('Gender vs Loan Status')
plt.show()

sns.countplot(df['Gender'],hue=df['Married'],palette='Set1')
print(pd.crosstab(df['Gender'],df['Married']))
plt.savefig('Gender vs Married')
plt.show()

sns.countplot(df['Gender'],hue=df['Self_Employed'],palette='Set1')
print(pd.crosstab(df['Gender'],df['Self_Employed']))
plt.savefig('Gender vs Self Employed')
plt.show()

sns.countplot(df['Gender'],hue=df['Property_Area'],palette='Set1')
print(pd.crosstab(df['Gender'],df['Property_Area']))
plt.savefig('Gender vs Property Area')
plt.show()



print(df['Loan_Status'].value_counts(), '\n' )
df['Loan_Status'].replace('Y',1,inplace=True)
df['Loan_Status'].replace('N',0,inplace=True)
print(df['Loan_Status'].value_counts(), '\n' )



print(df['Gender'].value_counts(), '\n' )
df.Gender=df.Gender.map({'Male':1,'Female':0})
print(df['Gender'].value_counts(), '\n' )


print(df['Married'].value_counts(), '\n' )
df.Married=df.Married.map({'Yes':1,'No':0})
print(df['Married'].value_counts(), '\n' )


print(df['Education'].value_counts(), '\n' )
df.Education=df.Education.map({'Graduate':1,'Not Graduate':0})
print(df['Education'].value_counts(), '\n' )


print(df['Self_Employed'].value_counts(), '\n' )
df.Self_Employed=df.Self_Employed.map({'Yes':1,'No':0})
print(df['Self_Employed'].value_counts(), '\n' )

print(df['Property_Area'].value_counts(), '\n' )
df.Property_Area=df.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
print(df['Property_Area'].value_counts(), '\n' )

print( df.head(5), '\n' )

print("Preprocessing Stage, Visualization Level 2 Done")


X = df.iloc[:,1:12]
#print("X = ",'\n', X)
y = df.iloc[:,-1]
#print('\n', Y)


train_X,test_X,train_y,test_y = train_test_split(
    X,y,test_size=0.1, random_state=seed )


print('\n train_X = \n', train_X)
print('\n test_X = \n', test_X)
print('\n train_y = \n', train_y)
print('\n test_y = \n', test_y)
final_test = test_X.iloc[0:15]
print('\n final_test = \n', final_test)


print("\n Machine Learning Model Build \n")
models=[]
models.append(("logreg",LogisticRegression()))


LR = LogisticRegression()
LR.fit(train_X,train_y)
pred = LR.predict(test_X)

print("accuracy_score", accuracy_score(test_y,pred), '\n')


outp = LR.predict(final_test)
print( "Output Prediction = ", outp, '\n' )
print('\n', test_y[:15])


print("Project End")






