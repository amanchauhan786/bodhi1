import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data_df = pd.read_csv("fetal_health (1).csv")

# Define features and target variable
X = data_df.drop(["fetal_health"], axis=1)
y = data_df["fetal_health"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=25)

# Initialize and train the model
gbcl = GradientBoostingClassifier(learning_rate=0.1,
                                  n_estimators=500, max_depth=6,
                                  min_samples_split=0.5, min_samples_leaf=0.3,
                                  max_features="log2", random_state=25)
gbcl.fit(X_train, y_train)

# Function to take input values from the user
def take_user_input():
    input_data = []
    st.write("Please enter the following features:")
    # Iterate through each feature except 'fetal_health'
    for feature in X.columns:
        value = st.number_input(feature)
        input_data.append(float(value))
    return np.array(input_data).reshape(1, -1)  # Reshape to match model input format

# Main Streamlit code
st.title("Fetal Healthcare Prediction 🏥🏥🏥")

# Set the background color
st.markdown(
    """
    <style>
    body {
        background-color: #FFE4E1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#Take input from the user
user_input = take_user_input()

# Make prediction
prediction = gbcl.predict(user_input)
st.write("# Predicted fetal distress:", prediction[0])

# Plotting a graph to display the prediction
st.write("Probability Distribution:")
plt.figure(figsize=(8, 6))
# Bar plot for the predicted fetal distress
plt.bar(["Normal", "Suspect", "Pathological"], gbcl.predict_proba(user_input)[0], color=['green', 'orange', 'red'])
plt.xlabel('Fetal Health')
plt.ylabel('Probability')
plt.title('Fetal Health Prediction')
st.pyplot(plt)

# Heatmap of predictions
st.write("# Heatmap of Predictions:")
predictions = gbcl.predict(X_test)
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Suspect', 'Pathological'], yticklabels=['Normal', 'Suspect', 'Pathological'])
plt.xlabel(' Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
st.pyplot(plt)

# Large text about fetal distress for each parameter value
st.write("# Fetal Distress for Each Parameter Value")
st.write("""
    In fetal healthcare prediction, it's essential to understand the impact of various parameters on fetal distress.
    Here's an overview:
    - **Baseline Fetal Heart Rate (FHR)**: Normal range is around 120 to 160 beats per minute (bpm). Higher or lower values may indicate distress.
    - **Accelerations**: A positive sign, indicating good fetal health.
    - **Decelerations**: Variable decelerations may signal fetal distress.
    - **Fetal Movement**: Decreased fetal movement could indicate distress.
    - **Uterine Contraction**: Normal contractions occur every 2-3 minutes and last around 60 seconds.
    - **Light Decelerations**: Occasional and temporary decreases in heart rate are normal.
    - **Severe Decelerations**: Indicate significant fetal distress and may require immediate intervention.
""")
st.markdown("** In future We can also add the feature of computer vision to calculate the values and feed it automatically and predict the fetal distress more easily. **")
def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url("https://www.istockphoto.com/vector/a-woman-with-a-child-in-her-arms-asks-herself-many-questions-conceptual-gm1214316850-353249950");
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
set_bg_hack_url()