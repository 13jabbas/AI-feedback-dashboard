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

# Dropdown selection for entity
st.subheader("Select Entity to View Metrics")
selected_entity = st.selectbox("Choose an entity", columns_to_evaluate)

# Display the metrics for the selected entity as gauges
st.subheader(f"Metrics Gauges for {selected_entity}")
metrics = results[selected_entity]

# Create gauges for each metric using Plotly
fig_gauges = make_subplots(rows=1, cols=4, subplot_titles=["Micro F1 Score", "Precision", "Recall", "AUC"], specs=[[{'type': 'indicator'}] * 4])

# Set dark mode colors for gauges
gauge_colors = {
    'bg_color': '#0e1117',
    'bar_color': {
        'Micro F1 Score': 'darkblue',
        'Precision': 'green',
        'Recall': 'orange',
        'AUC': 'red'
    }
}

fig_gauges.add_trace(go.Indicator(
    mode="gauge+number",
    value=metrics['Micro F1 Score'] * 100,
    title={'text': "Micro F1 Score", 'font': {'color': 'white'}},
    gauge={'axis': {'range': [0, 100], 'tickcolor': 'white'}, 'bar': {'color': gauge_colors['bar_color']['Micro F1 Score']}, 'bgcolor': gauge_colors['bg_color']}
), row=1, col=1)

fig_gauges.add_trace(go.Indicator(
    mode="gauge+number",
    value=metrics['Precision'] * 100,
    title={'text': "Precision", 'font': {'color': 'white'}},
    gauge={'axis': {'range': [0, 100], 'tickcolor': 'white'}, 'bar': {'color': gauge_colors['bar_color']['Precision']}, 'bgcolor': gauge_colors['bg_color']}
), row=1, col=2)

fig_gauges.add_trace(go.Indicator(
    mode="gauge+number",
    value=metrics['Recall'] * 100,
    title={'text': "Recall", 'font': {'color': 'white'}},
    gauge={'axis': {'range': [0, 100], 'tickcolor': 'white'}, 'bar': {'color': gauge_colors['bar_color']['Recall']}, 'bgcolor': gauge_colors['bg_color']}
), row=1, col=3)

fig_gauges.add_trace(go.Indicator(
    mode="gauge+number",
    value=metrics['AUC'] * 100,
    title={'text': "AUC", 'font': {'color': 'white'}},
    gauge={'axis': {'range': [0, 100], 'tickcolor': 'white'}, 'bar': {'color': gauge_colors['bar_color']['AUC']}, 'bgcolor': gauge_colors['bg_color']}
), row=1, col=4)

fig_gauges.update_layout(height=400, width=1200, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', title_text=f"Accuracy Gauges for {selected_entity}", font={'color': 'white'})
st.plotly_chart(fig_gauges)

# Display ROC curve for selected entity
st.subheader("ROC Curve")
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(metrics['FPR'], metrics['TPR'], label=f"{selected_entity} (AUC = {metrics['AUC']:.2f})", color='cyan')
ax.plot([0, 1], [0, 1], 'k--', label="Random Guessing (AUC = 0.50)", color='grey')
ax.set_facecolor('#0e1117')
ax.set_xlabel('False Positive Rate', color='white')
ax.set_ylabel('True Positive Rate', color='white')
ax.set_title(f'ROC Curve for {selected_entity}', color='white')
ax.legend(loc='lower right', facecolor='#262730')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.tick_params(colors='white')
st.pyplot(fig)

# Word Cloud for Hallucinations with dark theme
st.subheader("Word Cloud for Hallucinations")
new_df = hallucinations[hallucinations['Keyword Match Count'] == 0]
text = ' '.join(new_df['Description'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Set2').generate(text)
fig_wordcloud, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig_wordcloud)
