

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


#from wordcloud import WordCloud, STOPWORDS
from PIL import Image

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_columns = None

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('Crop_recommendation.csv')
df.head()
df.shape
df.info()
df.describe()

#correln between the features of the dataset

fig, ax = plt.subplots(1, 1, figsize=(15, 9))
numdf = df.drop('label',axis=1)
sns.heatmap(numdf.corr(), annot=True)
ax.set(xlabel='features')
ax.set(ylabel='features')

plt.title('Correlation between different features', fontsize = 20, c='black')
plt.show()

#we start building the model here

#from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

target = ['label']
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

label_encoder = LabelEncoder()
numerical_target = label_encoder.fit_transform(df[target])

print(df[['label']])
print(numerical_target)
df['numerical_label']=numerical_target
# Mapping back from numerical to original labels
decoded_labels = label_encoder.inverse_transform(numerical_target)
print("Decoded Labels:", decoded_labels)

X = df[features]
y = df[['numerical_label']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

models = []
models.append(('LogisticRegression', LogisticRegression(random_state=0)))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier(random_state=0)))
models.append(('XGBClassifier', XGBClassifier(random_state=0)))
models.append(('GradientBoostingClassifier', GradientBoostingClassifier(random_state=0)))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier(random_state=0)))

model_name = []
accuracy = []

for name, model in models: 
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    model_name.append(name)
    accuracy.append(metrics.accuracy_score(y_test,y_pred))
    print(name, metrics.accuracy_score(y_test,y_pred))

plt.figure(figsize=(15,9))
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Model')
sns.barplot(x = accuracy, y = model_name)
plt.show()

model=RandomForestClassifier(random_state=0)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn import metrics

cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(15,15))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Confusion Matrix - score:'+str(metrics.accuracy_score(y_test,y_pred))
plt.title(all_sample_title, size = 15)
plt.show()
print(metrics.classification_report(y_test,y_pred))

#  [code] {"execution":{"iopub.status.busy":"2024-12-06T19:56:40.369411Z","iopub.execute_input":"2024-12-06T19:56:40.369845Z","iopub.status.idle":"2024-12-06T19:57:15.307363Z","shell.execute_reply.started":"2024-12-06T19:56:40.369808Z","shell.execute_reply":"2024-12-06T19:57:15.306203Z"}}
final_model = RandomForestClassifier(random_state=0)
final_model.fit(X,y)

# Function to get user input  
def get_user_input():  
    print("Please enter the following environmental parameters:")  
    N = float(input("Nitrogen (N) content in soil (kg/ha): "))  
    P = float(input("Phosphorus (P) content in soil (kg/ha): "))  
    K = float(input("Potassium (K) content in soil (kg/ha): "))  
    temperature = float(input("Temperature (°C): "))  
    humidity = float(input("Humidity (%): "))  
    ph = float(input("pH level of the soil: "))  
    rainfall = float(input("Rainfall (mm): "))  
    return [N, P, K, temperature, humidity, ph, rainfall]  

# Get user input  
user_input = get_user_input()  

# Convert to DataFrame for model prediction  
user_input_df = pd.DataFrame([user_input], columns=features)  

# Predict the crop  
predicted_label_numeric = final_model.predict(user_input_df)  
predicted_label = label_encoder.inverse_transform(predicted_label_numeric)  

# Display the result  
print(f"The recommended crop for the given conditions is: {predicted_label[0]}") 


# Streamlit user interface  
st.title("Crop Recommendation System")  
st.write("Please enter the following environmental parameters:")  

# Collect user inputs  
N = st.number_input("Nitrogen (N) content in soil (kg/ha)", min_value=0.0)  
P = st.number_input("Phosphorus (P) content in soil (kg/ha)", min_value=0.0)  
K = st.number_input("Potassium (K) content in soil (kg/ha)", min_value=0.0)  
temperature = st.number_input("Temperature (°C)", min_value=0.0)  
humidity = st.number_input("Humidity (%)", min_value=0.0)  
ph = st.number_input("pH level of the soil", min_value=0.0)  
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)  

if st.button("Predict Crop"):  
    user_input = [[N, P, K, temperature, humidity, ph, rainfall]]  
    user_input_df = pd.DataFrame(user_input, columns=features)  
    
    # Predict the crop  
    predicted_label_numeric = model.predict(user_input_df)  
    predicted_label = label_encoder.inverse_transform(predicted_label_numeric)  
    
    # Display the result  
    st.success(f"The recommended crop for the given conditions is: {predicted_label[0]}")  
