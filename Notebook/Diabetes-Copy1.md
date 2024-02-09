---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

## Diabetes Predictor


Life Cycle of ML Project
- Problem Statement
- Data Collection
- Data Check
- EDA
- Data Pre-Processing
- Model Training
- Best Model


1) Problem Statement
- This project aims to create a program that a user can use to give them a prediction on whether or not they have diabetes.


2) Data Collection
- Dataset Source - https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
- Dataset has 9 columns and 100000 rows

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import nbformat
import warnings
%matplotlib inline
warnings.simplefilter(action='ignore', category=FutureWarning)
```

```python
df = pd.read_csv('Data/diabetes_prediction_dataset.csv')
```

```python
df.info()

```

2.2 Information
- gender: Sex of individuals -> Male/Female/Other
- age: age of the individual
- Hypertension: Whether or not the person has hypertension
- Smoking History: Never, No info, Current, Former, Ever, Not Current
- bmi: The Body Mass Index of the individual
- HbA1C Levels: The individuals HbA1c Levels
- Blood Glucos Level
- Diabetes: Are they diabetic

```python
df.isna().sum()
```

```python
df.drop_duplicates(inplace = True)
print('Number of Duplicates: {:,}'.format(df.duplicated().sum()))

```

```python
sns.countplot(data=df, x='gender')
plt.title('Gender Distribution')
df['gender'].value_counts()

```

```python
sns.countplot(data=df, x='smoking_history')
plt.title('Distribution of Smokers')
df['smoking_history'].value_counts()
```

```python
df = df[~(df['gender']=='Other')]
```

```python
dfcopy = df.loc[df['diabetes'].isin([1])]

dfcopy.head()

```

# EDA 

```python
# Define conditions based on gender and diabetes status
conditions = [
    (df['gender'] == 'Male') & (df['diabetes'] == 0),
    (df['gender'] == 'Male') & (df['diabetes'] == 1),
    (df['gender'] == 'Female') & (df['diabetes'] == 0),
    (df['gender'] == 'Female') & (df['diabetes'] == 1)
]

# Define corresponding choices for each condition
choices = [
    'Male | Normal',
    'Male | Diabetic',
    'Female | Normal',
    'Female | Diabetic'
]

# Apply conditions and choices to create the 'category' column
df['category'] = np.select(conditions, choices, default = 'Unknown')
```

```python
#Goal: create a plot that shows the correlation between diabetes and age
fig_age = px.box(
    df,
    x = 'diabetes',
    y = 'age',
    title = "Diabetes and Age",
    labels = {'diabetes': 'Condition', 'age': 'Age'},
    color=('gender'))

fig_age.update_layout(title_x = 0.5, legend_title = '')
fig_age.update_xaxes(tickvals = [0, 1], ticktext = ['Normal', 'Diabetic'])



```

As shown with through the plots above, the chance of having diabetes increases as one ages. Also in this case there is not much difference between males and females. The chance of having diabetes later in life is the same in both.

```python
import plotly.graph_objects as go
```

```python

fig_bmi_age = px.histogram(
    df,
    y= 'bmi',
    x='age', 
    title = "BMI and Age",
    histfunc='avg',
    barmode='overlay',
    color = 'category',
    color_discrete_map = {
        'Male | Normal': 'blue', 
        'Male | Diabetic': 'orange',
        'Female | Normal': 'lightgreen', 
        'Female | Diabetic': 'darkred'},
)

fig_bmi_age.show()

df=df.drop(columns='category')
```

Insights:
BMI tends to be highest in the age range 30-60 in all groups. In the diabetic groups, females tend to have higher BMIs.

```python
df1 = df[df['diabetes'] == 0]
df2 = df[df['diabetes']==1]


fig = px.pie(df1, names='smoking_history', color = 'smoking_history', color_discrete_map={'No info':'gray',
                                 'never':'red',
                                 'current':'royalblue',
                                 'former':'green',
                                 'not current' : 'orange',
                                 ' ever' : 'lightblue'},
             title='Smoking History and Not Diabetic')
fig1 = px.pie(df2, names='smoking_history', color = 'smoking_history', color_discrete_map={'No info':'gray',
                                 'never':'red',
                                 'current':'royalblue',
                                 'former':'green',
                                 'not current' : 'orange',
                                 ' ever' : 'lightblue'},
              title ='Smoking History and Diabetic')
fig.update_layout(height=400, width=600, showlegend=True)
fig.show()
fig1.update_layout(height= 400, width=600, showlegend=True)
fig1.show()

```

From the piecharts above, it can be seen that there seems to be a slight increase in the number of smokers when looking at diabetics as compared with non-diabetics.

```python
#diabetes and blood glucose 

fig_blood_glu = px.box(
    df, 
    x = 'diabetes',
    y = 'blood_glucose_level',
    color = 'gender',
    title = 'Blood Glucose Level by Diabetes',
    labels = {'diabetes': 'Condition', 'blood_glucose_level': 'Blood Glucose Level'})
fig_blood_glu.update_layout(title_x = 0.5, legend_title = '')

fig_blood_glu.update_xaxes(tickvals = [0, 1], ticktext = ['Normal', 'Diabetic'])

fig_blood_glu.show()
```

Insights:
- Blood Glucose Levels are generally higher in diabetics.

```python
diabetes = df[df['diabetes']==1]
diabetes = diabetes.drop(columns=['gender','age','smoking_history','bmi','HbA1c_level','blood_glucose_level'])
hyper_diabetes_y = diabetes.drop(columns=['heart_disease','diabetes'])

no_diabetes = df[df['diabetes']==0]
no_diabetes = no_diabetes.drop(columns=['gender','age','smoking_history','bmi','HbA1c_level','blood_glucose_level'])
hyper_diabetes_N = no_diabetes.drop(columns=['heart_disease','diabetes'])

counts_n = hyper_diabetes_N['hypertension'].value_counts().tolist()
values_n = hyper_diabetes_N['hypertension'].value_counts().index.tolist()

counts_y = hyper_diabetes_y['hypertension'].value_counts().tolist()
values_y = hyper_diabetes_y['hypertension'].value_counts().index.tolist()

fig_hyper_n = px.bar( 
    hyper_diabetes_N,
    x = values_n,
    y = counts_n,
    title = 'Hypertension and Non-Diabetics',
    color =["blue", "red"])

fig_hyper_n.update_layout(
    title_x = 0.5,
    xaxis_title = "Individuals withouth Diabetes who have or do not have Hypertension",
    yaxis_title = "Number of Each"
                       )

fig_hyper_n.update_xaxes(tickvals = [0, 1], ticktext = ['NO', 'YES'])

fig_hyper_n.layout.showlegend = False

fig_hyper_n.show()

y_percentage = 5375/len(hyper_diabetes_N)*100
n_percentage = 82271/len(hyper_diabetes_N)*100

print("The percentage non-diabetics with hypertension:", y_percentage)
print("The percentage non-diabetics without hypertension:", n_percentage)



fig_hyper_y = px.bar( 
    hyper_diabetes_y,
    x = values_y,
    y = counts_y,
    title = 'Hypertension and Diabetes',
    color =["blue", "red"])

fig_hyper_y.update_layout(
    title_x = 0.5,
    xaxis_title = "Individuals with Diabetes who have or do not have Hypertension",
    yaxis_title = "Number of Each"
                       )

fig_hyper_y.update_xaxes(tickvals = [0, 1], ticktext = ['NO', 'YES'])

fig_hyper_y.layout.showlegend = False

fig_hyper_y.show()

Y_percentage = 2086/len(hyper_diabetes)*100
N_percentage = 6396/len(hyper_diabetes)*100

print("The percentage diabetics with hypertension:", Y_percentage)
print("The percentage diabetics without hypertension:", N_percentage)
```

As you can see by looking at the percentages hypertension seems to be more prevelant in those with diabetes.

```python
heart_diabetes_y = diabetes.drop(columns=['hypertension','diabetes'])


heart_diabetes_N = no_diabetes.drop(columns=['hypertension','diabetes'])

counts_hn = heart_diabetes_N['heart_disease'].value_counts().tolist()
values_hn = heart_diabetes_N['heart_disease'].value_counts().index.tolist()

counts_hy = heart_diabetes_y['heart_disease'].value_counts().tolist()
values_hy = heart_diabetes_y['heart_disease'].value_counts().index.tolist()

fig_heart_n = px.bar( 
    heart_diabetes_N,
    x = values_hn,
    y = counts_hn,
    title = 'Heart Disease and Non-Diabetics',
    color =["Blue", "Red"])

fig_heart_n.update_layout(
    title_x = 0.5,
    xaxis_title = "Individuals withouth Diabetes who have or do not have Heart Disease",
    yaxis_title = "Number of Each"
                       )

fig_heart_n.update_xaxes(tickvals = [0, 1], ticktext = ['NO', 'YES'])

fig_heart_n.layout.showlegend = False

fig_heart_n.show()

yh_percentage = 2656/len(heart_diabetes_N)*100
nh_percentage = 84990/len(heart_diabetes_N)*100

print("The percentage non-diabetics with heart disease:", yh_percentage)
print("The percentage non-diabetics without heart disease:", nh_percentage)



fig_heart_y = px.bar( 
    heart_diabetes_y,
    x = values_hy,
    y = counts_hy,
    title = 'Heart Disease and Diabetes',
    color =["Blue", "Red"])

fig_heart_y.update_layout(
    title_x = 0.5,
    xaxis_title = "Individuals with Diabetes who have or do not have Heart Disease",
    yaxis_title = "Number of Each"
                       )

fig_heart_y.update_xaxes(tickvals = [0, 1], ticktext = ['NO', 'YES'])

fig_heart_y.layout.showlegend = False

fig_heart_y.show()

Yh_percentage = 1267/len(hyper_diabetes)*100
Nh_percentage = 7215/len(hyper_diabetes)*100

print("The percentage diabetics with Heart Disease:", Yh_percentage)
print("The percentage diabetics without Heart Disease:", Nh_percentage)
```

# Data Preprocession

```python
df["diabetes"].unique()
```

```python
df["diabetes"].value_counts()
```

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, minmax_scale 

```

```python
label_encoder = LabelEncoder()
print(label_encoder)
```

```python
X = df.iloc[:,0:9]
X = X.dropna()
print(type(X))

Y = df.iloc[:,-1]

Y
```

```python
X['gender'] = label_encoder.fit_transform(X["gender"])
X
```

```python
# OneHotEncoding

features = ['smoking_history']

encoder = OneHotEncoder()

X_encode = pd.DataFrame(encoder.fit_transform(X[features]).toarray(),
                          columns=encoder.get_feature_names_out(features))

X = pd.concat([X.drop(features, axis=1), X_encode], axis=1)

X.shape

```

```python
# Normaliztion 
sc_x = StandardScaler()
cols = ['age', 'bmi', 'HbA1c_level','blood_glucose_level']
X[cols] = sc_x.fit_transform(X[cols])
```

```python
Corr_map = X.corr()
plt.figure(figsize=(16,12))
sns.heatmap(Corr_map, annot=True, cmap='rocket')
plt.title('Correlation Matrix')
plt.show()
```

```python
from sklearn.utils import resample
df['gender'] = label_encoder.fit_transform(df['gender'])
df['smoking_history'] = label_encoder.fit_transform(df['smoking_history'])

x = df.drop(columns = ['diabetes', 'category']).values
y = df['diabetes'].values
from sklearn.model_selection import train_test_split

```

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=42)
```

```python
train_data = pd.concat([pd.DataFrame(x_train, columns = df.drop(columns = ['diabetes', 'category']).columns), pd.Series(y_train, name = 'target')],
                       axis = 1)
majority_class = train_data[train_data['target'] == 0]
minority_class = train_data[train_data['target'] == 1]

# Upsample the minority class
minority_upsampled = resample(minority_class, replace = True, n_samples = len(majority_class), random_state = 42)

# Combine majority class with upsampled minority class
balanced_data = pd.concat([majority_class, minority_upsampled])
print(balanced_data)
# Separate features and target variable for the balanced dataset
X_balanced = balanced_data.drop('target', axis = 1)
y_balanced = balanced_data['target']
print(X_balanced)

balanced_data.to_csv('Balanced.csv')
```

## Decision Tree

```python
from sklearn.ensemble import RandomForestClassifier
```

```python
rfc = RandomForestClassifier(n_estimators=200)
```

```python
rfc.fit(X_balanced,y_balanced)
```

```python
predictions = rfc.predict(x_test)
```

```python
from sklearn.metrics import classification_report,confusion_matrix
```

```python
print(classification_report(y_test, predictions))
```

```python
print(confusion_matrix(y_test,predictions))
```

# Web Application

```python
import streamlit as st
from PIL import Image

st.write(""" #Screening Application for Diabetes""")

image = Image.open('c:/Users/ryan_/Downloads/diabetes-528678_640.jpg')
st.image(image, use_column_width=True)
```

```python
def get_user_input():
    gender = st.sidebar.slider('Sex', )
    age = st.sidebar.slider('Your Age', )
    
```

```python

```

```python
df.info()
```

```python

```

```python

```
