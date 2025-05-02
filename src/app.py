from utils import db_connect
engine = db_connect()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load the data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/rosinni/k-nearest-neighbors-project-tutorial/refs/heads/main/winequality-red.csv"
    df = pd.read_csv(url, sep=";")
    df['label'] = df['quality'].apply(lambda q: 0 if q <= 4 else (1 if q <= 6 else 2))
    return df

df = load_data()
st.title("🍷 Wine Quality Classifier with KNN")
st.write("Predict wine quality using chemical properties and the K-Nearest Neighbors algorithm.")

# 2. Prepare data
X = df.drop(["label", "quality"], axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. K selection
st.sidebar.header("Model Configuration")
k = st.sidebar.slider("Choose the number of neighbors (k)", 1, 20, 5)

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

st.write(f"**Accuracy with k={k}:** {acc:.2f}")

# 4. Plot accuracy vs. k
@st.cache_data
def get_k_accuracies():
    k_range = range(1, 21)
    accs = []
    for i in k_range:
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train_scaled, y_train)
        y_k_pred = model.predict(X_test_scaled)
        accs.append(accuracy_score(y_test, y_k_pred))
    return list(k_range), accs

st.subheader("📈 Accuracy vs. k")
k_range, accuracies = get_k_accuracies()
fig, ax = plt.subplots()
ax.plot(k_range, accuracies, marker='o')
ax.set_xlabel("k")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs. Value of k")
st.pyplot(fig)

# 5. User input for prediction
st.subheader("🔮 Predict Wine Quality")
st.write("Enter the chemical properties of the wine below:")

input_features = []
feature_names = X.columns
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, format="%.4f")
    input_features.append(value)

if st.button("Predict"):
    input_scaled = scaler.transform([input_features])
    prediction = model.predict(input_scaled)[0]
    quality = ["Low", "Medium", "High"][prediction]
    st.success(f"This wine is likely of **{quality}** quality 🍷")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and scikit-learn.")

