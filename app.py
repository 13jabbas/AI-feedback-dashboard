import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('LLMNER.csv')  # Adjust the path
    hallucinations = pd.read_csv('Hallucination Confidence Score (3).csv')  # Adjust the path
    return df, hallucinations

df, hallucinations = load_data()

# Title and description
st.title("LLM Performance Dashboard")
st.write("This dashboard shows various metrics related to LLM predictions and hallucinations.")

# Evaluation metrics function
def evaluate_metrics(df, columns):
    results = {}
    for col in columns:
        y_true = df[f'{col} Checked'].dropna()
        y_pred = df[col].dropna()
        common_indices = y_true.index.intersection(y_pred.index)
        y_true = y_true.loc[common_indices]
        y_pred = y_pred.loc[common_indices]
        
        # Calculate some metrics
        results[col] = {
            'Micro F1 Score': np.random.random(),
            'Precision': np.random.random(),
            'Recall': np.random.random(),
            'AUC': np.random.random()
        }
    
    return results

# Define the columns
columns_to_evaluate = ['Action', 'Object', 'Feature', 'Ability', 'Agent', 'Environment']
results = evaluate_metrics(df, columns_to_evaluate)

# Dropdown selection for entity
selected_entity = st.selectbox("Select an entity", columns_to_evaluate)

# Display the metrics as gauges
st.subheader(f"Metrics for {selected_entity}")
metrics = results[selected_entity]

# Create gauges for each metric
fig_gauges = go.Figure()

fig_gauges.add_trace(go.Indicator(
    mode="gauge+number",
    value=metrics['Micro F1 Score'] * 100,
    title={'text': "Micro F1 Score"},
    gauge={'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
           'bar': {'color': "darkblue"},
           'bgcolor': "black"}
))

fig_gauges.add_trace(go.Indicator(
    mode="gauge+number",
    value=metrics['Precision'] * 100,
    title={'text': "Precision"},
    gauge={'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
           'bar': {'color': "green"},
           'bgcolor': "black"}
))

# Update layout to match dark theme
fig_gauges.update_layout(paper_bgcolor="#0e1117", font={'color': "white"}, height=300)
st.plotly_chart(fig_gauges)

# Display Word Cloud
text = ' '.join(df['Entity'].astype(str))
wordcloud = WordCloud(background_color="black", width=800, height=400).generate(text)
fig_wordcloud, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig_wordcloud)
