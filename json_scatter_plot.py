import plotly.graph_objects as go
import json

# Load the data from the JSON file
with open('prediction_results_on_non_shakespearean_texts.json', 'r') as file:
    data = json.load(file)

# Extracting data for plotting
x = [item['avg_raw_scores'][0] for item in data]
y = [item['avg_raw_scores'][1] for item in data]
colors = ['green' if item['prediction'] == 'good' else 'red' for item in data]
titles = [item['play_title'] for item in data]

# Create a scatter plot
fig = go.Figure(data=go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(color=colors, size=5),
    text=titles
))

# Set plot layout
fig.update_layout(title='Play Analysis', xaxis_title='Score 1', yaxis_title='Score 2')

# Save the figure to an HTML file
fig.write_html("scatter_plot_play_analysis.html")
