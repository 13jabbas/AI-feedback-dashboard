# PACKAGES

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize, LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(layout="wide")

# SIDE BAR LLM VERSION SELECTOR

# Create a toggle button for showing/hiding the sidebar content
show_sidebar = st.checkbox("Show LLM Versions", value=False)

if show_sidebar:
    with st.sidebar:
        st.header("")
        # Add a dropdown selector with the specified options
        llm_version = st.selectbox("Select LLM Version", ["LLM V1", "LLM V2", "LLM V3"])
        st.write(f"You selected: {llm_version}")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('LLMNER (1).csv')  # Modify path as needed
    hallucinations = pd.read_csv('Hallucination Confidence Score (3).csv')  # Modify path as needed
    return df, hallucinations

df, hallucinations = load_data()

# Columns to evaluate
columns_to_evaluate = ['Action', 'Object', 'Feature', 'Ability', 'Agent', 'Environment', 'Valence']

# Dictionary to store the results
results = {}

for col in columns_to_evaluate:
    # True labels and predicted labels, ensuring NaNs are handled
    y_true = df[f'{col} Checked'].dropna()
    y_pred = df[col].dropna()

    # Get the common indices to ensure lengths match
    common_indices = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[common_indices]
    y_pred = y_pred.loc[common_indices]

    # Ensure the lengths match after aligning
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch between true and predicted labels for {col}")

    # Combine y_true and y_pred before fitting LabelEncoder
    combined_labels = pd.concat([y_true, y_pred])
    le = LabelEncoder()
    le.fit(combined_labels)

    y_true_encoded = le.transform(y_true)
    y_pred_encoded = le.transform(y_pred)

    # Calculate metrics
    macro_f1 = f1_score(y_true_encoded, y_pred_encoded, average='macro')
    precision = precision_score(y_true_encoded, y_pred_encoded, average='macro')
    recall = recall_score(y_true_encoded, y_pred_encoded, average='macro')

    # Store results
    results[col] = {
        'Macro F1 Score': macro_f1,
        'Precision (Macro)': precision,
        'Recall (Macro)': recall
    }

# Dropdown for selecting entity
selected_entity = st.selectbox("Select Entity", columns_to_evaluate)

# Display metrics for the selected entity
if selected_entity in results:
    metrics = results[selected_entity]

    # Create gauge cluster
    fig = go.Figure()

    # F1 Score Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics['Macro F1 Score'],
        title={'text': "F1 Score"},
        gauge={'axis': {'range': [0, 1]}},
        domain={'x': [0, 0.33], 'y': [0, 1]}
    ))

    # Precision Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics['Precision (Macro)'],
        title={'text': "Precision"},
        gauge={'axis': {'range': [0, 1]}},
        domain={'x': [0.34, 0.67], 'y': [0, 1]}
    ))

    # Recall Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics['Recall (Macro)'],
        title={'text': "Recall"},
        gauge={'axis': {'range': [0, 1]}},
        domain={'x': [0.68, 1], 'y': [0, 1]}
    ))

    # Update layout
    fig.update_layout(title_text=f"Metrics for {selected_entity}", height=300)

    # Display the gauge cluster
    st.plotly_chart(fig)

# Create a DataFrame for ranking metrics
ranked_metrics = pd.DataFrame(results).T
ranked_metrics['Average'] = ranked_metrics[['Macro F1 Score', 'Precision (Macro)', 'Recall (Macro)']].mean(axis=1)
ranked_metrics = ranked_metrics.sort_values(by='Average', ascending=False)

# Leaderboard display
st.subheader("Entity Leaderboard")
for index, row in ranked_metrics.iterrows():
    st.write(f"**{index}:** Average Score = {row['Average']:.2f}")

# Plot the ROC curves for each entity
for col in columns_to_evaluate:
    if 'Checked' in df.columns:
        y_true = df[f'{col} Checked'].dropna()
        y_pred = df[col].dropna()

        common_indices = y_true.index.intersection(y_pred.index)
        y_true = y_true.loc[common_indices]
        y_pred = y_pred.loc[common_indices]

        # Binarize the output
        y_true_binarized = label_binarize(y_true, classes=le.classes_)
        y_pred_binarized = label_binarize(y_pred, classes=le.classes_)

        # Compute ROC AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(le.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
            roc_auc[i] = roc_auc_score(y_true_binarized[:, i], y_pred_binarized[:, i])

        # Plotting the ROC curves
        plt.figure(figsize=(12, 8))
        for i in range(len(le.classes_)):
            plt.plot(fpr[i], tpr[i], label=f"Class {le.classes_[i]} (AUC = {roc_auc[i]:.2f})")

        plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing (AUC = 0.50)")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for {col}')
        plt.legend(loc='lower right')
        plt.grid()

        # Display the ROC curve
        st.pyplot(plt)

##HALLUCINATION CONFIDENCE SCORES 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
import streamlit as st

# Load the CSV data
df = pd.read_csv('/content/drive/MyDrive/Capstone/StreamlitRoc.csv')

# Define ground truth and predicted probabilities
y_true = df['hallucination_groundtruth']  # Ground truth (1 for hallucination, 0 for non-hallucination)
y_scores = df['Hallucination Confidence Score']  # Predicted probabilities or scores

# Compute False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Calculate the Area Under the Curve (AUC)
roc_auc = roc_auc_score(y_true, y_scores)

# Compute F1 score for each threshold
f1_scores = []
for threshold in thresholds:
    y_pred = (y_scores >= threshold).astype(int)  # Convert probabilities to binary predictions
    f1 = f1_score(y_true, y_pred)
    f1_scores.append(f1)

# Find the maximum F1 score
max_f1 = max(f1_scores)

# Find all thresholds that give the maximum F1 score
optimal_indices = [i for i, f1 in enumerate(f1_scores) if f1 == max_f1]
optimal_thresholds = thresholds[optimal_indices]

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='b')

# Mark the points of the optimal thresholds on the ROC curve
for idx in optimal_indices:
    plt.scatter(fpr[idx], tpr[idx], color='red', label=f'Optimal Threshold ({thresholds[idx]:.2f})')

# Add labels and title
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve (TPR vs FPR)')
plt.legend()

# Display the plot in Streamlit
st.pyplot(plt)

# Display the optimal thresholds and maximum F1 score in Streamlit
st.write(f'Optimal Thresholds: {optimal_thresholds}')
st.write(f'Maximum F1 Score: {max_f1:.4f}')
