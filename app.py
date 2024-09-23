
# PACKAGES

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize, LabelEncoder

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
    df = pd.read_csv('LLMNER.csv')  # Modify path as needed
    hallucinations = pd.read_csv('Hallucination Confidence Score (3).csv')  # Modify path as needed
    return df, hallucinations

df, hallucinations = load_data()

# Columns to evaluate
columns_to_evaluate = ['Action', 'Object', 'Feature', 'Ability', 'Agent', 'Environment']

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
    st.metric(label="F1 Score", value=f"{metrics['Macro F1 Score']:.2f}")
    st.metric(label="Precision", value=f"{metrics['Precision (Macro)']:.2f}")
    st.metric(label="Recall", value=f"{metrics['Recall (Macro)']:.2f}")

    # Optionally, display gauges using st.progress for a visual representation
    st.progress(metrics['Macro F1 Score'], text="F1 Score")
    st.progress(metrics['Precision (Macro)'], text="Precision")
    st.progress(metrics['Recall (Macro)'], text="Recall")

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

# Create a DataFrame for the table
table_data = {
    'Column': [],
    'Macro F1 Score': [],
    'Precision (Macro)': [],
    'Recall (Macro)': []
}

for col, metrics in results.items():
    table_data['Column'].append(col)
    table_data['Macro F1 Score'].append(metrics['Macro F1 Score'])
    table_data['Precision (Macro)'].append(metrics['Precision (Macro)'])
    table_data['Recall (Macro)'].append(metrics['Recall (Macro)'])

# Convert to DataFrame for display
results_df = pd.DataFrame(table_data)

# Display the results table
st.dataframe(results_df)



#HEATMAP Display 

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Read the CSV file
df = pd.read_csv('Hallucination Confidence Score (3).csv')

# Convert 'Hallucination Confidence Score' from string percentage to float
df['Hallucination Confidence Score'] = df['Hallucination Confidence Score'].str.rstrip('%').astype('float') / 100

# Function to create the gauge
def create_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,  # Convert to percentage for display
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': 'rgba(0,0,0,0)'},  # Hide the default bar
            'steps': [
                {'range': [0, 20], 'color': 'green'},
                {'range': [20, 40], 'color': 'yellowgreen'},
                {'range': [40, 60], 'color': 'yellow'},
                {'range': [60, 80], 'color': 'orange'},
                {'range': [80, 100], 'color': 'red'}
            ],
        },
        number={'suffix': "%"}
    ))
    fig.update_layout(height=150, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Display the dataframe with interactive features
confidence_threshold = st.slider("Filter by confidence score", 0.0, 1.0, 0.0)
filtered_df = df[df['Hallucination Confidence Score'] >= confidence_threshold]

# Pagination
page_size = 10
total_pages = (len(filtered_df) - 1) // page_size + 1

# Ensure total_pages is at least 1
if total_pages < 1:
    total_pages = 1

# Reset page number to 1 if confidence threshold changes
page = st.session_state.get('page', 1)
if 'confidence_threshold_value' not in st.session_state or st.session_state['confidence_threshold_value'] != confidence_threshold:
    st.session_state['confidence_threshold_value'] = confidence_threshold
    page = 1
st.session_state['page'] = st.number_input("Page", min_value=1, max_value=total_pages, step=1, value=page)

# Calculate start and end index for the current page
start_idx = (page - 1) * page_size
end_idx = start_idx + page_size

# Display only the relevant slice of the dataframe
paginated_df = filtered_df.iloc[start_idx:end_idx]

# Loop through the paginated dataframe and display content
for i, row in paginated_df.iterrows():
    st.subheader(f"Review {start_idx + i + 1}")
    st.write(f"**Review:** {row['Review Text Original']}")
    st.write(f"**Annotation:** {row['Annotated Text']}")
    st.write(f"**Description:** {row['Description Original']}")
    
    # Add the gauge for the confidence score
    st.plotly_chart(create_gauge(row['Hallucination Confidence Score']), use_container_width=True)
    st.markdown("---")
