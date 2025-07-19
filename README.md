# BDD100K Object Detection Dashboard

This repository provides a parser and dashboard for visualizing and analyzing object category counts, bounding box statistics, image-level attributes, and bounding box anomalies from BDD100K label files.

## Project Structure

- `src/bdd100k_bbox_parser.py`: Parser for extracting bounding boxes, attributes, and exporting to CSV.
- `src/dashboard.py`: Dash app for visualizing statistics and anomalies using precomputed CSVs.
- `bdd100k_labels/100k/train/` and `bdd100k_labels/100k/val/`: Directories containing BDD100K label JSON files.
- `bdd100k_images_100k/`: Directory containing BDD100K images (not tracked by git).
- `train.csv`, `val.csv`: Precomputed CSVs with all relevant object and image-level information (not tracked by git).

## Requirements

- Docker (recommended for easy setup)
- Alternatively: Python 3.8+, `pip install -r requirements.txt`

## Quick Start with Docker

1. **Clone this repository and place your BDD100K label files in `bdd100k_labels/100k/train/` and `bdd100k_labels/100k/val/`. Place images in `bdd100k_images_100k/`.**

2. **Build the Docker image:**
   ```sh
   docker build -t bdd100k-dashboard .
   ```

3. **Run the Docker container:**
   ```sh
   docker run -p 8050:8050 -v $(pwd)/bdd100k_labels:/app/bdd100k_labels -v $(pwd)/bdd100k_images_100k:/app/bdd100k_images_100k bdd100k-dashboard
   ```
   - On Windows (PowerShell), use:
     ```sh
     docker run -p 8050:8050 -v ${PWD}/bdd100k_labels:/app/bdd100k_labels -v ${PWD}/bdd100k_images_100k:/app/bdd100k_images_100k bdd100k-dashboard
     ```

4. **Open your browser and go to:**  
   [http://localhost:8050](http://localhost:8050)

## Development (without Docker)

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

2. Run the dashboard:
   ```sh
   python src/dashboard.py
   ```

## How it works

- On first run, the code parses all JSON label files and creates `train.csv` and `val.csv` with all object and image-level information.
- On subsequent runs, the code checks for new images and only updates the CSVs if new data is present.
- All dashboard plots and anomaly visualizations are generated from these CSVs for fast loading and analysis.

## Features

- **Category Distribution:** Bar plots comparing object counts per category for train and val splits.
- **Image Attribute Distributions:** Grouped bar plots for weather, scene, and time of day.
- **Average Objects per Image:** Bar plots showing average number of objects per image for each category.
- **Bounding Box Area Distribution:** Box plots of bounding box area per category for train and val.
- **Bounding Box Anomalies:** Visualization of outlier bounding boxes (by area) drawn on images, with category and area labels.

## Notes

- The dashboard will show the count of each object category (e.g., car, person, traffic light, etc.) in your dataset, as well as bounding box and attribute statistics.
- Adjust the `limit` parameter in the parser methods if you want to process more or fewer files.
- The following files and folders are **not tracked by git** (see `.gitignore`):  
  - `bdd100k_images_100k/`
  - `bdd100k_images_100k.zip`
  - `train.csv`
  - `val.csv`

