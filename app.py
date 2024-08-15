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
