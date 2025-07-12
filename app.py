
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nbformat as nbf
import os
import time
import uuid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_supervised(df, model_name, target_column):
    df = df.copy()
    for col in df.select_dtypes(include='object'):
        df[col] = LabelEncoder().fit_transform(df[col])
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "Support Vector Machine":
        model = SVC()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return acc, cm, classification_report(y_test, preds), model

def train_unsupervised(df, model_name):
    df = df.copy()
    for col in df.select_dtypes(include='object'):
        df[col] = LabelEncoder().fit_transform(df[col])
    X = df.values
    if model_name == "KMeans Clustering":
        model = KMeans(n_clusters=3)
        labels = model.fit_predict(X)
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='Set1')
        plt.title("KMeans Clustering")
        plt.savefig("cluster_plot.png")
        return labels, "cluster_plot.png"
    elif model_name == "DBSCAN":
        model = DBSCAN()
        labels = model.fit_predict(X)
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='Set1')
        plt.title("DBSCAN Clustering")
        plt.savefig("dbscan_plot.png")
        return labels, "dbscan_plot.png"
    elif model_name == "PCA":
        model = PCA(n_components=2)
        pca_data = model.fit_transform(X)
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1])
        plt.title("PCA Projection")
        plt.savefig("pca_plot.png")
        return pca_data, "pca_plot.png"

def train_reinforcement():
    states = 5
    actions = 2
    Q = np.zeros((states, actions))
    for episode in range(100):
        state = np.random.randint(0, states)
        action = np.argmax(Q[state])
        reward = np.random.rand()
        Q[state, action] += 0.1 * (reward + 0.9 * np.max(Q[state]) - Q[state, action])
    return Q

def generate_python_code(model_name, task_type, target_column):
    code = f"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('your_dataset.csv')
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=['{target_column}'])
y = df['{target_column}']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""
    if model_name == "Logistic Regression":
        code += "\nfrom sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression()"
    elif model_name == "Decision Tree":
        code += "\nfrom sklearn.tree import DecisionTreeClassifier\nmodel = DecisionTreeClassifier()"
    elif model_name == "Random Forest":
        code += "\nfrom sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier()"
    elif model_name == "Support Vector Machine":
        code += "\nfrom sklearn.svm import SVC\nmodel = SVC()"
    code += """
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))
"""
    return code

def generate_unsupervised_code(model_name):
    base = """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('your_dataset.csv')
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.values
"""
    if model_name == "KMeans Clustering":
        return base + """
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("KMeans Clustering")
plt.show()
"""
    elif model_name == "DBSCAN":
        return base + """
from sklearn.cluster import DBSCAN
model = DBSCAN()
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma')
plt.title("DBSCAN Clustering")
plt.show()
"""
    elif model_name == "PCA":
        return base + """
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA Projection")
plt.show()
"""

def generate_reinforcement_code():
    return """
import numpy as np

states = 5
actions = 2
Q = np.zeros((states, actions))

for episode in range(100):
    state = np.random.randint(0, states)
    action = np.argmax(Q[state])
    reward = np.random.rand()
    Q[state, action] += 0.1 * (reward + 0.9 * np.max(Q[state]) - Q[state, action])

print("Trained Q-table:")
print(Q)
"""

def generate_ipynb(code_string):
    nb = nbf.v4.new_notebook()
    code_cell = nbf.v4.new_code_cell(code_string)
    nb['cells'] = [code_cell]
    with open("generated_notebook.ipynb", "w") as f:
        nbf.write(nb, f)

st.set_page_config(page_title="🧠 ML Code Studio", layout="centered")
st.title("🧠 ML Code Studio")
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    task_type = st.selectbox("🧠 Select ML Task", ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"])

    if task_type == "Supervised Learning":
        model_name = st.selectbox("🔍 Choose Model", ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"])
        file_key = uploaded_file.name.replace(".", "_") if uploaded_file else "default_file"
        target_column = st.selectbox("🎯 Select Target Column", df.columns, key=f"target_selector_{file_key}")

        if st.button("🚀 Train Model"):
            acc, cm, report, trained_model = train_supervised(df, model_name, target_column)
            st.success(f"✅ Accuracy: {acc:.2f}")
            st.text(report)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            code = generate_python_code(model_name, task_type, target_column)
            with open("generated_code.py", "w") as f:
                f.write(code)
                f.flush()
            generate_ipynb(code)
            time.sleep(1)
            with open("generated_code.py", "rb") as f:
                st.download_button("⬇️ Download Python Script", f, file_name="model_code.py")
            with open("generated_notebook.ipynb", "rb") as f:
                st.download_button("⬇️ Download Notebook", f, file_name="model_code.ipynb")

    elif task_type == "Unsupervised Learning":
        model_name = st.selectbox("🔍 Choose Model", ["KMeans Clustering", "DBSCAN", "PCA"])
        if st.button("🚀 Train Model"):
            labels, plot_path = train_unsupervised(df, model_name)
            st.success("✅ Model trained")
            st.image(plot_path)
            code = generate_unsupervised_code(model_name)
            with open("generated_unsup.py", "w") as f:
                f.write(code)
                f.flush()
            generate_ipynb(code)
            time.sleep(1)
            with open("generated_unsup.py", "rb") as f:
                st.download_button("⬇️ Download Python Script", f, file_name="unsupervised_code.py")
            with open("generated_notebook.ipynb", "rb") as f:
                st.download_button("⬇️ Download Notebook", f, file_name="unsupervised_code.ipynb")

    elif task_type == "Reinforcement Learning":
        model_name = st.selectbox("🔍 Choose Algorithm", ["Q-Learning (Simulated)"])
        if st.button("🚀 Simulate Training"):
            Q = train_reinforcement()
            st.success("✅ Q-Learning Done")
            st.dataframe(Q)
            code = generate_reinforcement_code()
            with open("generated_rl.py", "w") as f:
                f.write(code)
                f.flush()
            generate_ipynb(code)
            time.sleep(1)
            with open("generated_rl.py", "rb") as f:
                st.download_button("⬇️ Download Python Script", f, file_name="q_learning.py")
            with open("generated_notebook.ipynb", "rb") as f:
                st.download_button("⬇️ Download Notebook", f, file_name="q_learning.ipynb")
