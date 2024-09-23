
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

# Instead of plt.show(), use st.pyplot
st.pyplot(plt)

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
st.dataframe(results_df)  # Use st.dataframe instead of print

# Optionally display the table visually
# If you want a visual representation of the table as a matplotlib figure
fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size
ax.axis('off')
ax.axis('tight')
ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')

# Again, use st.pyplot
st.pyplot(fig)

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
