import plotly.graph_objects as go
import json

# Load the data from the JSON file
with open('prediction_results_on_non_shakespearean_texts.json', 'r') as file:
    data = json.load(file)

# Compute the average scores for each entry
for item in data:
    item['avg_score'] = sum(item['avg_raw_scores']) / 2

# Separate, sort and then combine "bad" and "good" predictions
bad_data = sorted([item for item in data if item['prediction'] == 'bad'], key=lambda x: x['avg_score'], reverse=True)
good_data = sorted([item for item in data if item['prediction'] == 'good'], key=lambda x: x['avg_score'], reverse=True)
combined_data = bad_data + good_data

# Colors for each bar
colors = ['red' if item['prediction'] == 'bad' else 'green' for item in combined_data]

# Create the bar plot
fig = go.Figure(go.Bar(
    y=[item['avg_score'] for item in combined_data],
    marker_color=colors,
    hoverinfo='text+y',
    text=[item['play_title'] for item in combined_data],
    orientation='v'
))

# Set plot layout
fig.update_layout(
    title='Average Score per Play Grouped by Prediction',
    yaxis_title='Average Score',
    xaxis_title='Titles',
    xaxis_showticklabels=False
)

# Save the figure to an HTML file
fig.write_html("grouped_sorted_bar_plot_play_analysis.html")
