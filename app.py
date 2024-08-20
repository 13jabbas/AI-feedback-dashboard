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


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('LLMNER.csv')  # Modify path as needed
    hallucinations = pd.read_csv('Hallucination Confidence Score (3).csv')  # Modify path as needed
    return df, hallucinations

df, hallucinations = load_data()

# Streamlit Title and description
st.title("LLM Performance Dashboard")

# Layout for the entities section with metrics
st.header("Entity Metrics")

# Create a three-column layout
col1, col2, col3 = st.columns([1, 4, 2])  # col1 for the dropdown, col2 for gauges and ROC curve, col3 for entities list

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

# Sidebar for dropdown selection
with col1:
    st.subheader("Select Entity to View Metrics")
    selected_entity = st.selectbox("Choose an entity", columns_to_evaluate)

# Display the metrics for the selected entity as gauges
with col2:
    st.subheader(f"Metrics Gauges for {selected_entity}")
    metrics = results[selected_entity]

    # Create gauges for each metric using Plotly
    fig_gauges = make_subplots(rows=1, cols=4, subplot_titles=["Micro F1 Score", "Precision", "Recall", "AUC"], specs=[[{'type': 'indicator'}] * 4])

    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics['Micro F1 Score'] * 100,
        title={'text': "Micro F1 Score"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"}}
    ), row=1, col=1)

    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics['Precision'] * 100,
        title={'text': "Precision"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
    ), row=1, col=2)

    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics['Recall'] * 100,
        title={'text': "Recall"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "orange"}}
    ), row=1, col=3)

    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics['AUC'] * 100,
        title={'text': "AUC"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red"}}
    ), row=1, col=4)

    fig_gauges.update_layout(height=400, width=1200, title_text=f"Accuracy Gauges for {selected_entity}")
    st.plotly_chart(fig_gauges)

   

# Add a section for displaying entities ordered by accuracy in the third column
with col3:
    st.subheader("Entities Ordered by Accuracy")

    # Add headings for 'Entity' and 'Accuracy'
    st.markdown("<b>Entity</b> | <b>Accuracy</b>", unsafe_allow_html=True)

    # Sort entities based on their AUC score in descending order
    sorted_entities = sorted(results.items(), key=lambda x: x[1]['AUC'], reverse=True)

    # Display the entities in a table-like layout
    for entity, metrics in sorted_entities:
        entity_name = f"{entity}"
        accuracy = f"{metrics['AUC']:.2f}"

        # Create a table-like structure with columns
        col_ent, col_acc, col_bar = st.columns([3, 2, 4])
        col_ent.write(entity_name)
        col_acc.write(accuracy)
        col_bar.progress(int(metrics['AUC'] * 100))  # Multiplied by 100 to match Streamlit's 0-100 scale

# Add another section for hallucinations
st.header("Hallucinations Analysis")

# Word Cloud for Hallucinations
st.subheader("Word Cloud for Hallucinations")
new_df = hallucinations[hallucinations['Keyword Match Count'] == 0]
text = ' '.join(new_df['Description'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='BuPu').generate(text)
fig_wordcloud, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig_wordcloud)

# Bigram analysis
st.subheader("Top 10 Most Common Bigrams")

# Generate bigrams
vectorizer = CountVectorizer(ngram_range=(2, 2))
X = vectorizer.fit_transform(new_df['Description'].astype(str))
bigrams = vectorizer.get_feature_names_out()
bigram_counts = X.sum(axis=0).A1
bigram_freq = dict(zip(bigrams, bigram_counts))
sorted_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)[:10]

# Display top 10 bigrams
bigram_df = pd.DataFrame(sorted_bigrams, columns=['Bigram', 'Frequency'])
st.dataframe(bigram_df)

#HEATMAP Display 


import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# Sample DataFrame with string percentages

df = pd.read_csv('Hallucination Confidence Score (3).csv')


# Convert 'Hallucination Confidence Score' from string percentage to float
df['Hallucination Confidence Score'] = df['Hallucination Confidence Score'].str.rstrip('%').astype('float') 

# Create a pivot table for the heatmap
heatmap_data = df.pivot(
    index='Description Original',
    columns='Review Text Original',
    values='Hallucination Confidence Score'
)

# Ensure that the pivot table has no NaNs by filling with zeros or another appropriate value
heatmap_data = heatmap_data.fillna(0)

# Create the heatmap with hover data
fig = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,  # 2D array of heatmap values
    x=heatmap_data.columns,  # Review Text Original as x-axis
    y=heatmap_data.index,  # Description Original as y-axis
    text=heatmap_data.apply(lambda row: [f"Review: {review}<br>Description: {desc}<br>Score: {score:.2f}" 
                                        for review, score in zip(row.index, row.values)], axis=1).tolist(),
    hoverinfo="text",
    colorscale='Viridis',  # Color gradient
    colorbar=dict(title="Confidence Score"),  # Colorbar label
    zmin=0,  # Optional: Set minimum value for color scale
    zmax=1   # Optional: Set maximum value for color scale
))

# Update layout to ensure square cells
fig.update_layout(
    title='Interactive Heatmap of Hallucination Confidence Scores',
    xaxis_title='Review Text Original',
    yaxis_title='Description Original',
    xaxis=dict(
        tickangle=-45,  # Rotate x-axis labels for better readability
        ticks='',  # Remove ticks
        showticklabels=True,  # Show x-axis labels
        scaleanchor='y',  # Lock aspect ratio of x-axis to y-axis
        scaleratio=1  # Ensure squares by setting equal scaling
    ),
    yaxis=dict(
        ticks='',  # Remove ticks
        showticklabels=True,  # Show y-axis labels
        scaleanchor='x',  # Lock aspect ratio of y-axis to x-axis
        scaleratio=1  # Ensure squares by setting equal scaling
    ),
    autosize=False,
    width=1200,  # Adjust width to fit more data
    height=1200,  # Adjust height to ensure cells are square
    dragmode='zoom'  # Enable zoom and pan functionality
)

# Display the heatmap in Streamlit
tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    st.plotly_chart(fig, theme="streamlit")
with tab2:
    st.plotly_chart(fig, theme=None)




# Display HCS
HCS_df = df['Hallucination Confidence Score']
st.dataframe(HCS_df)


