import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("modelData2.csv",encoding='iso-8859-1')
#print(df.head())

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df
df = handle_non_numerical_data(df)
#print(df.head())


X = df[['OCCUPATION','IF_OCCUPATION_HAZARDOUS','GENDER','AGE']]
y = df['STATE_OF_PROPOSAL']

df = pd.DataFrame(df)
df=df.iloc[:,:].values
STATE_OF_PROPOSAL=LabelEncoder()
df[:,4]=STATE_OF_PROPOSAL.fit_transform(df[:,4])
#df[:,0]=STATE_OF_PROPOSAL.fit_transform(df[:,0])
#print(df)

#print(f'X : {X.shape}')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.20, random_state=101)

print(f'X_train : {X_train.shape}')
print(f'y_train : {y_train.shape}')
print(f'X_test : {X_test.shape}')
print(f'y_test : {y_test.shape}')

rf_Model = RandomForestClassifier(oob_score=True)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

rf_Model.fit(X_train,y_train)

rf_Model.oob_score_

print (f'Train Accuracy - : {rf_Model.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {rf_Model.score(X_test,y_test):.3f}')


classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

#pickle.dump(classifier, open("model.pkl", "wb"))