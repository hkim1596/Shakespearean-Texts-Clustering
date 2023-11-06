import json
import plotly.offline as offline
import plotly.graph_objects as go

# 1. Reading the JSON file
with open("prediction_results_on_non_shakespearean_texts.json", "r") as file:
    data = json.load(file)

# Sort the data by the 'prediction' values
data = sorted(data, key=lambda x: x['prediction'])

titles = [item["play_title"] for item in data]
predictions = [item["prediction"] for item in data]

# 2. Creating the Plotly table
fig = go.Figure(data=[go.Table(
    header=dict(values=['Play Title', 'Prediction']),
    cells=dict(values=[titles, predictions]),
    columnwidth=[500, 100]  # Adjusting the column widths; modify as needed
)])

# 3. Saving the table as an HTML file
offline.plot(fig, filename='sorted_prediction_results_table.html', auto_open=False)
