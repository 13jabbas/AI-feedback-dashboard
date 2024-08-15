# Importing necessary libraries
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

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('LLMNER.csv')  # Modify path as needed
    hallucinations = pd.read_csv('Hallucination Confidence Score (3).csv')  # Modify path as needed
    return df, hallucinations

df, hallucinations = load_data()

# Page Layout
st.set_page_config(layout="wide")

# Streamlit Title and description
st.title("LLM Performance Dashboard")

### Sidebar for entity selection ###
st.sidebar.title("Entity Selection")
columns_to_evaluate = ['Action', 'Object', 'Feature', 'Ability', 'Agent', 'Environment']
selected_entity = st.sidebar.selectbox("Choose an entity", columns_to_evaluate)

### Section 1: Entities and Metrics ###
st.markdown("## Entity Metrics")

# Micro F1 score, precision, recall, and ROC evaluation for each attribute
def evaluate_metrics(df, columns_to_evaluate):
    results = {}
    for col in columns_to_evaluate:
        y_true = df[f'{col} Checked'].dropna()
        y_pred = df[col].dropna()
        common_indices = y_true.index.intersection(y_pred.index)
        y_true = y_true.loc[common_indices]
        y_pred = y_pred.loc[common_indices]
        
        le = LabelEncoder()
        combined_labels = pd.concat([y_true, y_pred])
        le.fit(combined_labels)
        y_true_encoded = le.transform(y_true)
        y_pred_encoded = le.transform(y_pred)

        micro_f1 = f1_score(y_true_encoded, y_pred_encoded, average='micro')
        precision = precision_score(y_true_encoded, y_pred_encoded, average='micro')
        recall = recall_score(y_true_encoded, y_pred_encoded, average='micro')
        classes = le.classes_
        y_true_binarized = label_binarize(y_true_encoded, classes=range(len(classes)))
        y_pred_binarized = label_binarize(y_pred_encoded, classes=range(len(classes)))

        if y_true_binarized.shape[1] == 1:
            auc = roc_auc_score(y_true_encoded, y_pred_encoded)
            fpr, tpr, _ = roc_curve(y_true_encoded, y_pred_encoded)
        else:
            auc = roc_auc_score(y_true_binarized, y_pred_binarized, average='micro')
            fpr, tpr, _ = roc_curve(y_true_binarized.ravel(), y_pred_binarized.ravel())

        results[col] = {'Micro F1 Score': micro_f1, 'Precision': precision, 'Recall': recall, 'AUC': auc, 'FPR': fpr, 'TPR': tpr}
    
    return results

# Metrics evaluation
results = evaluate_metrics(df, columns_to_evaluate)
metrics = results[selected_entity]

# Display the metrics for the selected entity as gauges
st.subheader(f"Metrics Gauges for {selected_entity}")

# Create gauges for each metric using Plotly with box-like appearance
fig_gauges = make_subplots(rows=1, cols=4, 
                           subplot_titles=["Micro F1 Score", "Precision", "Recall", "AUC"],
                           specs=[[{'type': 'indicator'}] * 4])

# Add border and background to the gauges
def add_gauge_trace(fig, value, title, row, col, bar_color):
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
        }
    ), row=row, col=col)

# Add each gauge in its own "box"
add_gauge_trace(fig_gauges, metrics['Micro F1 Score'], "Micro F1 Score", 1, 1, "darkblue")
add_gauge_trace(fig_gauges, metrics['Precision'], "Precision", 1, 2, "green")
add_gauge_trace(fig_gauges, metrics['Recall'], "Recall", 1, 3, "orange")
add_gauge_trace(fig_gauges, metrics['AUC'], "AUC", 1, 4, "red")

# Update layout with better spacing and appearance
fig_gauges.update_layout(
    height=400, 
    width=1200, 
    margin=dict(l=20, r=20, t=40, b=20),
    paper_bgcolor="lightgray",  # Optional background color for the whole plot
    plot_bgcolor="white",
    title_text=f"Accuracy Gauges for {selected_entity}"
)

st.plotly_chart(fig_gauges)

# Display ROC curve for selected entity
st.subheader("ROC Curve")
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(metrics['FPR'], metrics['TPR'], label=f"{selected_entity} (AUC = {metrics['AUC']:.2f})")
ax.plot([0, 1], [0, 1], 'k--', label="Random Guessing (AUC = 0.50)")
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC Curve for {selected_entity}')
ax.legend(loc='lower right')
st.pyplot(fig)

### Section 2: Hallucinations ###
st.markdown("---")  # Divider
st.markdown("## Hallucinations Analysis")

# Word Cloud for Hallucinations
st.subheader("Word Cloud for Hallucinations")
new_df = hallucinations[hallucinations['Keyword Match Count'] == 0]
text = ' '.join(new_df['Description'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
fig_wordcloud, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig_wordcloud)
