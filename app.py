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
import streamlit as st

# Read the CSV file
df = pd.read_csv('Hallucination Confidence Score (3).csv')

# Convert 'Hallucination Confidence Score' from string percentage to float
df['Hallucination Confidence Score'] = df['Hallucination Confidence Score'].str.rstrip('%').astype('float') / 100

# Dropdown for selecting score range
range_options = {
    '0%': (0.0, 0.0),
    '10% - 20%': (0.10, 0.20),
    '20% - 30%': (0.20, 0.30),
    '30% - 40%': (0.30, 0.40),
    '40% - 50%': (0.40, 0.50),
    '50% - 60%': (0.50, 0.60),
    '60% - 70%': (0.60, 0.70),
    '70% - 80%': (0.70, 0.80),
    '80% - 90%': (0.80, 0.90),
    '90% - 100%': (0.90, 1.0)
}

# Display the dropdown in the sidebar
selected_range = st.selectbox('Select Hallucination Confidence Score Range:', list(range_options.keys()))

# Filter the dataframe based on the selected range
min_score, max_score = range_options[selected_range]
filtered_df = df[(df['Hallucination Confidence Score'] >= min_score) & (df['Hallucination Confidence Score'] <= max_score)]

# Set the page size to 5 reviews per page
page_size = 5

# Initialize session state for pagination (if it doesn't exist)
if 'page_number' not in st.session_state:
    st.session_state['page_number'] = 0

# Function to get the current page of reviews
def get_paginated_data(df, page_number, page_size):
    start_idx = page_number * page_size
    end_idx = start_idx + page_size
    return df.iloc[start_idx:end_idx]

# Function to display reviews
def display_reviews(df):
    for i, row in df.iterrows():
        st.subheader(f"Review {i + 1}")
        st.write(f"**Review:** {row['Review Text Original']}")
        st.write(f"**Annotation:** {row['Annotated Text']}")
        st.write(f"**Description:** {row['Description Original']}")

        # Highlighted and right-aligned Hallucination Confidence Score
        st.markdown(f"""
            <div style='text-align: right; font-size: 20px; color: red; font-weight: bold;'>
                Hallucination Confidence Score: {row['Hallucination Confidence Score'] * 100:.2f}%
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")

# Get the current page's data based on the filtered dataframe
paginated_df = get_paginated_data(filtered_df, st.session_state['page_number'], page_size)

# Display the current page of reviews
display_reviews(paginated_df)

# Add "Next" and "Previous" buttons for pagination
col1, col2, col3 = st.columns([1, 1, 1])

# Only show the "Previous" button if we're beyond the first page
if st.session_state['page_number'] > 0:
    if col1.button("Previous"):
        st.session_state['page_number'] -= 1

# Only show the "Next" button if there are more reviews to display
if len(filtered_df) > (st.session_state['page_number'] + 1) * page_size:
    if col3.button("Next"):
        st.session_state['page_number'] += 1



##ROC CURVE


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

# Load the CSV file
file_path = 'hallucinationThreshold (3).csv'
df = pd.read_csv(file_path)

# Convert 'Hallucination Confidence Score' to numeric if needed
df['Hallucination Confidence Score'] = pd.to_numeric(df['Hallucination Confidence Score'], errors='coerce')

# Drop rows where the 'Hallucination Confidence Score' is NaN (if any)
df = df.dropna(subset=['Hallucination Confidence Score'])

# Define ground truth and predicted scores
y_true = df['hallucination_groundtruth']  # Ground truth (1 for hallucination, 0 for non-hallucination)
y_scores = df['Hallucination Confidence Score']  # Predicted probabilities or confidence scores

# Calculate FPR, TPR, and thresholds for ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = roc_auc_score(y_true, y_scores)

# Calculate F1 score for each threshold
f1_scores = []
for threshold in thresholds:
    y_pred = (y_scores >= threshold).astype(int)  # Convert probabilities to binary predictions
    f1 = f1_score(y_true, y_pred)
    f1_scores.append(f1)

# Convert to numpy arrays for easier sorting
f1_scores = np.array(f1_scores)
thresholds = np.array(thresholds)

# Sort thresholds based on F1 scores in descending order
sorted_indices = np.argsort(f1_scores)[::-1]
sorted_thresholds = thresholds[sorted_indices]
sorted_f1_scores = f1_scores[sorted_indices]

# Get the top 10 optimal thresholds and their corresponding F1 scores
top_10_thresholds = sorted_thresholds[:10]
top_10_f1_scores = sorted_f1_scores[:10]

# Create the ROC plot for a selected threshold
def plot_roc_curve(fpr, tpr, roc_auc, chosen_threshold, chosen_f1, fpr_chosen, tpr_chosen):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line for random predictions
    
    # Mark the chosen threshold point on the ROC curve
    plt.scatter(fpr_chosen, tpr_chosen, label=f'Threshold {chosen_threshold:.2f} (F1 = {chosen_f1:.2f})', color='red')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Hallucination Detection')
    plt.legend(loc='lower right')
    plt.grid(True)
    return plt

# Display the section in Streamlit
st.title("Hallucination Detection Evaluation")
st.subheader("ROC Curve with Threshold Selection")

# Define the threshold ranges as percentages
threshold_ranges = [(i, i + 5) for i in range(0, 100, 5)]
threshold_options = [f'{start}-{end}%' for start, end in threshold_ranges]

# Dropdown menu to select a threshold range
threshold_option = st.selectbox("Select a Threshold Range", threshold_options)

# Get the indices of thresholds within the selected range
selected_range_index = threshold_options.index(threshold_option)
range_start, range_end = threshold_ranges[selected_range_index]
selected_indices = [i for i, threshold in enumerate(top_10_thresholds * 100) if range_start <= threshold < range_end]

# Ensure there are available thresholds within the selected range
if selected_indices:
    chosen_index = selected_indices[0]  # Select the first index in the range
    chosen_threshold = top_10_thresholds[chosen_index]
    chosen_f1 = top_10_f1_scores[chosen_index]
    fpr_chosen = fpr[sorted_indices[chosen_index]]
    tpr_chosen = tpr[sorted_indices[chosen_index]]
    
    # Plot and display the ROC curve with the selected threshold
    roc_plot = plot_roc_curve(fpr, tpr, roc_auc, chosen_threshold, chosen_f1, fpr_chosen, tpr_chosen)
    st.pyplot(roc_plot)
else:
    st.write("No thresholds available within the selected range.")
