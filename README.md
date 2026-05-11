# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, drop unnecessary columns, and encode categorical variables. 
2.Define the features (X) and target variable (y). 
3.Split the data into training and testing sets. 
4.Train the logistic regression model, make predictions, and evaluate using accuracy and other
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NITESH BHANDARI K
RegisterNumber:  212225240101
*/



import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from google.colab import files
uploaded = files.upload()  # A file picker will appear — select your Placement_Data.csv
import io
filename = list(uploaded.keys())[0]
data = pd.read_csv(io.BytesIO(uploaded[filename]))
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)
classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)



```

## Output:
HEAD:

<img width="1279" height="330" alt="image" src="https://github.com/user-attachments/assets/c39c60a5-3e9d-4b93-a664-05713938c963" />

COPY:

<img width="1141" height="345" alt="image" src="https://github.com/user-attachments/assets/d1b2aece-6117-4f46-8139-201abb70e63c" />

FIT TRANSFORM:

<img width="1122" height="707" alt="image" src="https://github.com/user-attachments/assets/87f973d2-fd66-4eb7-b1a4-3e8d9256581a" />

LOGISTIC REGRESSION:

<img width="1231" height="309" alt="image" src="https://github.com/user-attachments/assets/fec3c3d8-3736-4f27-a4de-557d32ba5dc9" />

ACCURACY SCORE, CONFUSION MATRIX, CLASSIFICATION REPORT & PREDICTION: 

<img width="1088" height="407" alt="image" src="https://github.com/user-attachments/assets/2656df38-fbcc-4b96-bd7b-09f219565fab" />




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
