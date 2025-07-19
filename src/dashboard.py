import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
from bdd100k_bbox_parser import BDD100KBoundingBoxParser
import numpy as np
import matplotlib.pyplot as plt
import cv2
import base64
from io import BytesIO

# --- Load or update CSVs ---
train_csv = "train.csv"
val_csv = "val.csv"
train_df = BDD100KBoundingBoxParser.load_or_update_csv('./bdd100k_labels/100k/train', train_csv)
val_df = BDD100KBoundingBoxParser.load_or_update_csv('./bdd100k_labels/100k/val', val_csv)

train_num_images = train_df['image_name'].nunique()
val_num_images = val_df['image_name'].nunique()

# --- Object category distribution ---
def get_category_counts(df):
    return df['category'].value_counts().to_dict()

train_category_counts = get_category_counts(train_df)
val_category_counts = get_category_counts(val_df)
all_categories = set(train_category_counts) | set(val_category_counts)
all_categories = sorted(
    all_categories,
    key=lambda c: train_category_counts.get(c, 0) + val_category_counts.get(c, 0),
    reverse=True
)
train_counts = [train_category_counts.get(c, 0) for c in all_categories]
val_counts = [val_category_counts.get(c, 0) for c in all_categories]
cat_fig = go.Figure([
    go.Bar(name='Train', x=all_categories, y=train_counts, marker_color='blue'),
    go.Bar(name='Val', x=all_categories, y=val_counts, marker_color='orange')
])
cat_fig.update_layout(
    barmode='group',
    title='Object Category Counts in BDD100K (Train vs Val)',
    xaxis_title='Category',
    yaxis_title='Count'
)

# --- Image-level attribute distributions ---
def make_attr_fig(attr_key, title):
    train_dist = train_df[f'img_attr_{attr_key}'].value_counts().to_dict()
    val_dist = val_df[f'img_attr_{attr_key}'].value_counts().to_dict()
    all_keys = sorted(set(train_dist) | set(val_dist))
    train_vals = [train_dist.get(k, 0) for k in all_keys]
    val_vals = [val_dist.get(k, 0) for k in all_keys]
    fig = go.Figure([
        go.Bar(name='Train', x=all_keys, y=train_vals, marker_color='blue'),
        go.Bar(name='Val', x=all_keys, y=val_vals, marker_color='orange')
    ])
    fig.update_layout(
        barmode='group',
        title=title,
        xaxis_title=attr_key.capitalize(),
        yaxis_title='Number of Images'
    )
    return fig
weather_fig = make_attr_fig('weather', 'Weather Distribution (Train vs Val)')
scene_fig = make_attr_fig('scene', 'Scene Distribution (Train vs Val)')
timeofday_fig = make_attr_fig('timeofday', 'Time of Day Distribution (Train vs Val)')

# --- Average objects per image ---
def avg_objects_per_image(df, num_images):
    return df.groupby('category')['image_name'].count() / num_images

train_avg_obj = avg_objects_per_image(train_df, train_num_images)
val_avg_obj = avg_objects_per_image(val_df, val_num_images)
all_cats = sorted(set(train_avg_obj.index) | set(val_avg_obj.index), key=lambda c: (train_avg_obj.get(c,0)+val_avg_obj.get(c,0)), reverse=True)
train_avg = [train_avg_obj.get(c, 0) for c in all_cats]
val_avg = [val_avg_obj.get(c, 0) for c in all_cats]
avg_obj_fig = go.Figure([
    go.Bar(name='Train', x=all_cats, y=train_avg, marker_color='blue'),
    go.Bar(name='Val', x=all_cats, y=val_avg, marker_color='orange')
])
avg_obj_fig.update_layout(
    barmode='group',
    title='Average Number of Objects per Image (per Category)',
    xaxis_title='Category',
    yaxis_title='Average Objects per Image'
)

# --- Bounding box area distribution per class (box plot) ---
def bbox_area_boxplot(df, split_name):
    data = []
    for cat in all_cats:
        areas = df[df['category'] == cat].eval('(bbox_x2-bbox_x1)*(bbox_y2-bbox_y1)').values
        if len(areas) > 0:
            data.append(go.Box(y=areas, name=cat, boxmean=True, boxpoints='outliers'))
    fig = go.Figure(data)
    fig.update_layout(
        title=f'Bounding Box Area Distribution per Category ({split_name})',
        xaxis_title='Category',
        yaxis_title='Bounding Box Area (pixels^2)',
        showlegend=False
    )
    return fig
train_bbox_area_fig = bbox_area_boxplot(train_df, "Train")
val_bbox_area_fig = bbox_area_boxplot(val_df, "Val")

def detect_bbox_anomalies(df, n_images=4):
    """
    Detect and visualize bounding box area anomalies using IQR per category.
    Returns a list of (image_name, image_bgr, bboxes) for n_images with anomalies.
    """
    anomalies = []
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat]
        areas = (cat_df['bbox_x2'] - cat_df['bbox_x1']) * (cat_df['bbox_y2'] - cat_df['bbox_y1'])
        q1 = np.percentile(areas, 25)
        q3 = np.percentile(areas, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (areas < lower) | (areas > upper)
        outliers = cat_df[outlier_mask]
        for _, row in outliers.iterrows():
            anomalies.append({
                'image_name': row['image_name'],
                'bbox': [row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']],
                'category': row['category'],
                'area': (row['bbox_x2'] - row['bbox_x1']) * (row['bbox_y2'] - row['bbox_y1'])
            })
    # Group by image and select up to n_images with anomalies
    img_to_boxes = {}
    for a in anomalies:
        img_to_boxes.setdefault(a['image_name'], []).append(a)
    selected = list(img_to_boxes.items())[:n_images]
    return selected

def draw_bboxes_on_image(image_path, bboxes):
    """
    Draw bounding boxes with category and area on the image.
    Returns the image as a base64-encoded PNG.
    """
    # Read image (handle missing images gracefully)
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        # Create a blank image if not found
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox['bbox'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{bbox['category']} ({int(bbox['area'])})"
        cv2.putText(img, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    # Convert to PNG base64
    buf = BytesIO()
    plt.imsave(buf, img, format='png')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64

# --- Anomaly detection and visualization ---
# You may need to adjust the path to your images
image_dir = './bdd100k_images_100k/100k/train'
anomaly_images = []
anomaly_info = detect_bbox_anomalies(train_df, n_images=4)
for image_name, bboxes in anomaly_info:
    img_path = f"{image_dir}/{image_name}.jpg"
    img_b64 = draw_bboxes_on_image(img_path, bboxes)
    anomaly_images.append((image_name, img_b64, bboxes))

# --- Dash Layout ---
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("BDD100K Dataset Dashboard"),
    html.P(f"Number of images in training set: {train_num_images}"),
    html.P(f"Number of images in validation set: {val_num_images}"),
    html.H2("Object Category Distribution (Train vs Val)"),
    dcc.Graph(figure=cat_fig),
    html.H2("Average Number of Objects per Image (per Category)"),
    dcc.Graph(figure=avg_obj_fig),
    html.H2("Bounding Box Area Distribution per Category (Train)"),
    dcc.Graph(figure=train_bbox_area_fig),
    html.H2("Bounding Box Area Distribution per Category (Val)"),
    dcc.Graph(figure=val_bbox_area_fig),
    html.H2("Image Attribute Distributions"),
    html.H3("Weather"),
    dcc.Graph(figure=weather_fig),
    html.H3("Scene"),
    dcc.Graph(figure=scene_fig),
    html.H3("Time of Day"),
    dcc.Graph(figure=timeofday_fig),
    html.H2("Bounding Box Area Anomalies (Train)"),
    html.Div([
        html.Div([
            html.H4(f"Image: {img_name}"),
            html.Img(src=f"data:image/png;base64,{img_b64}", style={"width": "400px"}),
            html.Ul([
                html.Li(f"{bbox['category']} area={int(bbox['area'])} [{int(bbox['bbox'][0])},{int(bbox['bbox'][1])},{int(bbox['bbox'][2])},{int(bbox['bbox'][3])}]")
                for bbox in bboxes
            ])
        ], style={"display": "inline-block", "margin": "20px", "vertical-align": "top"})
        for img_name, img_b64, bboxes in anomaly_images
    ])
])

if __name__ == "__main__":
    app.run(debug=True)
