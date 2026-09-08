import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import streamlit as st
import numpy as np

# Page setup
st.set_page_config(
    page_title="Decision Tree Classifier",
    page_icon="🌳",
    layout="wide"
)
st.title("Decision Tree Classification")
st.write("Water Potability")

# Load dataset
df = pd.read_csv("WP.csv")

st.subheader("DATASET")
st.dataframe(df)

# Handle missing values
df["ph"] = df["ph"].fillna(df["ph"].mean())
df["Sulfate"] = df["Sulfate"].fillna(df["Sulfate"].mean())
df["Trihalomethanes"] = df["Trihalomethanes"].fillna(df["Trihalomethanes"].mean())

# Target column (keep numeric labels but map later for visualization)
y = df["Potability"]
x = df[["ph", "Solids", "Sulfate"]]

# Scale features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42
)

# Sidebar settings
st.sidebar.header("Decision Tree Settings")
criterion = st.sidebar.selectbox("Select Criterion", ["gini", "entropy"])
max_depth = st.sidebar.slider("Maximum Tree Depth", min_value=1, max_value=10, value=3)

# Train model
dt_cl = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
model_dt = dt_cl.fit(x_train, y_train)

# Predictions
y_pred = model_dt.predict(x_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Performance")
st.metric("Accuracy", f"{accuracy:.2f}")

# Confusion Matrix
c_m = confusion_matrix(y_test, y_pred, labels=[0, 1])
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(
    c_m,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No-P", "P"],
    yticklabels=["No-P", "P"],
    ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Classification Report
st.subheader("Classification Report")
c_m_r = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(c_m_r).transpose()
st.data_editor(report_df)

# Decision Tree Visualization
st.subheader("Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(20, 20))
plot_tree(
    model_dt,
    feature_names=x.columns,
    class_names=[str(c) for c in model_dt.classes_],  # convert labels to strings
    filled=True,
    rounded=True,
    ax=ax
)
st.pyplot(fig)
