
# PACKAGES

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize, LabelEncoder
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

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

# Create a three-column layout
col1, col2, col3 = st.columns([1, 4, 2])  # col1 for the dropdown, col2 for gauges and ROC curve, col3 for entities list

import pandas as pd
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

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

    # Macro F1 score
    macro_f1 = f1_score(y_true_encoded, y_pred_encoded, average='macro')

    # Macro Precision and Recall
    precision = precision_score(y_true_encoded, y_pred_encoded, average='macro')
    recall = recall_score(y_true_encoded, y_pred_encoded, average='macro')

    # AUC
    # Label binarization is required for multiclass/multilabel AUC calculation
    classes = le.classes_
    y_true_binarized = label_binarize(y_true_encoded, classes=range(len(classes)))
    y_pred_binarized = label_binarize(y_pred_encoded, classes=range(len(classes)))

    # Handle binary and multi-class cases differently for AUC
    if y_true_binarized.shape[1] == 1:  # binary case
        auc = roc_auc_score(y_true_encoded, y_pred_encoded)
        fpr, tpr, _ = roc_curve(y_true_encoded, y_pred_encoded)
    else:  # multi-class case
        auc = roc_auc_score(y_true_binarized, y_pred_binarized, average='micro')
        fpr, tpr, _ = roc_curve(y_true_binarized.ravel(), y_pred_binarized.ravel())

    # Store results
    results[col] = {
        'Macro F1 Score': macro_f1,
        'Precision (Macro)': precision,
        'Recall (Macro)': recall,
        'AUC': auc,
        'FPR': fpr,
        'TPR': tpr
    }

# Plot the ROC curves
plt.figure(figsize=(12, 8))
for col, metrics in results.items():
    plt.plot(metrics['FPR'], metrics['TPR'], label=f"{col} (AUC = {metrics['AUC']:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing (AUC = 0.50)")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Create a DataFrame for the table
table_data = {
    'Column': [],
    'Macro F1 Score': [],
    'Precision (Macro)': [],
    'Recall (Macro)': [],
    'AUC': []
}

for col, metrics in results.items():
    table_data['Column'].append(col)
    table_data['Macro F1 Score'].append(metrics['Macro F1 Score'])
    table_data['Precision (Macro)'].append(metrics['Precision (Macro)'])
    table_data['Recall (Macro)'].append(metrics['Recall (Macro)'])
    table_data['AUC'].append(metrics['AUC'])

# Convert to DataFrame for display
results_df = pd.DataFrame(table_data)

# Display the results table
print(results_df)

# Optionally display the table visually with matplotlib
fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size
ax.axis('off')
ax.axis('tight')
ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')
plt.show()

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
