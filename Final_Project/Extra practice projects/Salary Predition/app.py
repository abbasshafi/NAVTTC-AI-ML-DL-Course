import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

string = "Salary Prediction Model"
st.set_page_config(page_title=string, page_icon="😀")

st.title("Welcome to Salary Predictior App.😀")

st.write("""
### Salary Prediction Model
Salary vs. Experience
""")

df = pd.read_csv("Salary_Data.csv")

X = df.iloc[:,[0]].values
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

exp = st.sidebar.slider("Experience",1,10,1)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict([[exp]])

st.write(f"Experience: ", exp)
st.write(f"Salary: ", float(y_pred))

fig = plt.figure()
plt.scatter(X_test,y_test, alpha=0.8, cmap='viridis')
plt.plot(X_train, reg.predict(X_train), color='m')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.colorbar()

st.pyplot(fig)





