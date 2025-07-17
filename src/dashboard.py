import dash
from dash import dcc, html
import plotly.express as px
from bdd100k_bbox_parser import BDD100KBoundingBoxParser

# Initialize parser and get category counts
parser = BDD100KBoundingBoxParser('./bdd100k_labels/100k/train')
category_counts = parser.count_categories(limit=100)  # Adjust limit as needed

# Prepare data for plotly
categories = list(category_counts.keys())
counts = list(category_counts.values())
fig = px.bar(x=categories, y=counts, labels={'x': 'Category', 'y': 'Count'},
             title='Object Category Counts in BDD100K')

# Dash app layout
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("BDD100K Object Category Dashboard"),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run(debug=True)
