# BDD100K Object Detection Dashboard

This repository provides a parser and dashboard for visualizing object category counts from BDD100K label files.

## Project Structure

- `src/bdd100k_bbox_parser.py`: Parser for extracting bounding boxes and category counts.
- `src/dashboard.py`: Dash app for visualizing object category counts.
- `bdd100k_labels/100k/train/`: Directory containing BDD100K label JSON files.

## Requirements

- Docker (recommended for easy setup)
- Alternatively: Python 3.8+, `pip install -r requirements.txt`

## Quick Start with Docker

1. **Clone this repository and place your BDD100K label files in `bdd100k_labels/100k/train/`.**

2. **Build the Docker image:**
   ```sh
   docker build -t bdd100k-dashboard .
   ```

3. **Run the Docker container:**
   ```sh
   docker run -p 8050:8050 -v $(pwd)/bdd100k_labels:/app/bdd100k_labels bdd100k-dashboard
   ```
   - On Windows (PowerShell), use:
     ```sh
     docker run -p 8050:8050 -v ${PWD}/bdd100k_labels:/app/bdd100k_labels bdd100k-dashboard
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

## Notes

- The dashboard will show the count of each object category (e.g., car, person, traffic light, etc.) in your dataset.
- Adjust the `limit` parameter in `dashboard.py` if you want to process more or fewer files.

