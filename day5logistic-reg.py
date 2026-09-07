import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

#page configuaration
st.set_page_config(
    page_title="LOGISTIC REGRESSION MODEL",
    page_icon="😊",
    layout="wide"
)
#load model
with open("LR_MODEL.pkl","rb")as file:
    model=pickle.load(file)
st.title("LOGISTIC REGRESSION")

st.subheader("DATA")
df=pd.read_csv("wp.csv")
st.dataframe(df.head(5))


st.write("Enter input values")
Hardness=st.number_input("Hardness",min_value=0.0,max_value=400.0,value=20.0)
Solids=st.number_input("Solids",min_value=0.0,max_value=65000.0,value=40.0)
Turbidity=st.number_input("Turbidity",min_value=0.0,max_value=10.0,value=5.0)
#prediction button
if st.button("Predict"):
    input_data=pd.DataFrame({"Hardness":[Hardness],"Solids":[Solids],"Turbidity":[Turbidity]})
    prediction=model.predict(input_data)
    if prediction[0]==1:
        st.error("prediction:Potability")
    else:
        st.success("prediction:No Potability")
st.subheader("CONFUSION MATRIX")
#feature selection
x=df[["Hardness","Solids","Turbidity"]]
y=df[["Potability"]]


#feature scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
#train test split
from sklearn.model_selection import train_test_split
x_scaled_train,x_scaled_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)
#model_prediction
y_pred=model.predict(x_scaled_test)
#model evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
accuracy=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy:",accuracy)
print("Classification_report:",cr)
print("Confusion_matrix:",cm)

st.dataframe(cm)
st.subheader("CONFUSION MATRIX DISPLAY")

disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
import matplotlib.pyplot as plt
fig, ax= plt.subplots(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Actual 0","Actual TrueNegative1"],
  
    ax=ax
)
ax.set_xlabel("predicted value")
ax.set_ylabel("actual value")
ax.set_title("confusion matrix")