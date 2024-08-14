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
from collections import Counter
from nltk import bigrams
import nltk

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('LLMNER.csv')  # Modify path as needed
    hallucinations = pd.read_csv('/path_to_data/Hallucination_Confidence_Score.csv')  # Modify path as needed
    return df, hallucinations

df, hallucinations = load_data()

# Streamlit Title and description
st.title("LLM Performance Dashboard")
st.write("This dashboard shows various metrics related to LLM predictions and hallucinations.")

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
columns_to_evaluate = ['Action', 'Object', 'Feature', 'Ability', 'Agent', 'Environment']
results = evaluate_metrics(df, columns_to_evaluate)

# Display ROC curves in Streamlit
st.subheader("ROC Curves")
fig, ax = plt.subplots(figsize=(12, 8))
for col, metrics in results.items():
    ax.plot(metrics['FPR'], metrics['TPR'], label=f"{col} (AUC = {metrics['AUC']:.2f})")
ax.plot([0, 1], [0, 1], 'k--', label="Random Guessing (AUC = 0.50)")
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(loc='lower right')
st.pyplot(fig)

# Display metric results
st.subheader("Evaluation Metrics")
for col, metrics in results.items():
    st.write(f"Metrics for {col}:")
    for metric, value in metrics.items():
        if metric not in ['FPR', 'TPR']:
            st.write(f"  {metric}: {value:.4f}")

# Accuracy Gauges for attributes
st.subheader("Accuracy Gauges for Attributes")

def calculate_accuracies(df):
    accuracies = {}
    for col in df.columns:
        if 'Checked' in col:
            original_col = col.replace(' Checked', '')
            correct_predictions = (df[col] == df[original_col]).sum()
            total_predictions = len(df[col])
            accuracy = correct_predictions / total_predictions
            accuracies[original_col] = accuracy
    return accuracies

accuracies = calculate_accuracies(df)
fig_gauges = make_subplots(rows=1, cols=len(accuracies), subplot_titles=list(accuracies.keys()), specs=[[{'type': 'indicator'}] * len(accuracies)])
for i, (attr, accuracy) in enumerate(accuracies.items()):
    fig_gauges.add_trace(go.Indicator(mode="gauge+number", value=accuracy * 100, title={'text': attr}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"}}), row=1, col=i+1)
fig_gauges.update_layout(height=400, width=1500, title_text="Accuracy Gauges for Attributes")
st.plotly_chart(fig_gauges)

# Word Cloud for Hallucinations
st.subheader("Word Cloud for Hallucinations")
new_df = hallucinations[hallucinations['Keyword Match Count'] == 0]
text = ' '.join(new_df['Description'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
fig_wordcloud, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig_wordcloud)
