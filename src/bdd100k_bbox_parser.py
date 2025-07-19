import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm


class BDD100KBoundingBoxParser:
    """
    Parser for BDD100K label JSON files to extract bounding box annotations.
    """

    def __init__(self, label_dir: str):
        """
        Initialize the parser with the directory containing label JSON files.

        Args:
            label_dir (str): Path to the directory with BDD100K label JSON files.
        """
        self.label_dir = Path(label_dir)

    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a single JSON file and extract bounding box annotations.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            List[Dict[str, Any]]: List of bounding box dictionaries.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        bboxes = []
        # Iterate over all frames and objects
        for frame in data.get('frames', []):
            for obj in frame.get('objects', []):
                if 'box2d' in obj:
                    bbox = {
                        'category': obj.get('category'),
                        'box2d': obj['box2d'],
                        'attributes': obj.get('attributes', {}),
                        'id': obj.get('id'),
                        # Add more fields if needed
                    }
                    bboxes.append(bbox)
        return bboxes

    def parse_all(self, limit: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        results = {}
        files = list(self.label_dir.glob('*.json'))
        print(f"[parse_all] Found {len(files)} files in {self.label_dir}")
        if limit:
            files = files[:limit]
            print(f"[parse_all] Limiting to first {limit} files")
        total = len(files)
        print("[parse_all] Progress:")
        for idx, file_path in enumerate(files):
            percent = ((idx + 1) / total) * 100
            if idx % 100 == 0 or idx == total - 1:
                print(f"\r  {percent:.2f}% completed", end="", flush=True)
            results[file_path.name] = self.parse_file(str(file_path))
        print()
        return results

    def count_categories(self, limit: Optional[int] = None) -> Dict[str, int]:
        category_counts = {}
        files = list(self.label_dir.glob('*.json'))
        print(f"[count_categories] Found {len(files)} files in {self.label_dir}")
        if limit:
            files = files[:limit]
            print(f"[count_categories] Limiting to first {limit} files")
        total = len(files)
        print("[count_categories] Progress:")
        for idx, file_path in enumerate(files):
            percent = ((idx + 1) / total) * 100
            if idx % 100 == 0 or idx == total - 1:
                print(f"\r  {percent:.2f}% completed", end="", flush=True)
            bboxes = self.parse_file(str(file_path))
            for bbox in bboxes:
                cat = bbox.get('category')
                if cat:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
        print()
        return category_counts

    def get_image_attributes(self, limit: Optional[int] = None) -> list:
        attr_list = []
        files = list(self.label_dir.glob('*.json'))
        print(f"[get_image_attributes] Found {len(files)} files in {self.label_dir}")
        if limit:
            files = files[:limit]
            print(f"[get_image_attributes] Limiting to first {limit} files")
        total = len(files)
        print("[get_image_attributes] Progress:")
        for idx, file_path in enumerate(files):
            percent = ((idx + 1) / total) * 100
            if idx % 100 == 0 or idx == total - 1:
                print(f"\r  {percent:.2f}% completed", end="", flush=True)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            attr = data.get('attributes', {})
            attr_list.append(attr)
        print()
        return attr_list

    def get_bbox_stats(self, limit: Optional[int] = None):
        from collections import defaultdict
        files = list(self.label_dir.glob('*.json'))
        print(f"[get_bbox_stats] Found {len(files)} files in {self.label_dir}")
        if limit:
            files = files[:limit]
            print(f"[get_bbox_stats] Limiting to first {limit} files")
        total = len(files)
        print("[get_bbox_stats] Progress:")
        per_image_counts = defaultdict(list)
        bbox_dims = defaultdict(list)
        for idx, file_path in enumerate(files):
            percent = ((idx + 1) / total) * 100
            if idx % 100 == 0 or idx == total - 1:
                print(f"\r  {percent:.2f}% completed", end="", flush=True)
            bboxes = self.parse_file(str(file_path))
            img_cat_count = defaultdict(int)
            for bbox in bboxes:
                cat = bbox.get('category')
                box = bbox.get('box2d')
                if cat and box:
                    w = box['x2'] - box['x1']
                    h = box['y2'] - box['y1']
                    area = w * h
                    bbox_dims[cat].append({'width': w, 'height': h, 'area': area})
                    img_cat_count[cat] += 1
            for cat, count in img_cat_count.items():
                per_image_counts[cat].append(count)
        print()
        return per_image_counts, bbox_dims

    @staticmethod
    def avg_objects_per_image(per_image_counts: dict, num_images: int) -> dict:
        """
        Compute average number of objects per image for each category.

        Args:
            per_image_counts (dict): category -> list of counts per image
            num_images (int): total number of images

        Returns:
            dict: category -> average objects per image
        """
        import numpy as np
        return {cat: np.sum(counts)/num_images for cat, counts in per_image_counts.items()}

    def export_objects_to_csv(self, csv_path: str, limit: Optional[int] = None):
        """
        Export all objects (with image and object attributes) to a CSV file.
        Only new images are processed if the CSV already exists.

        Args:
            csv_path (str): Path to the output CSV file.
            limit (Optional[int]): Maximum number of files to parse.
        """
        files = list(self.label_dir.glob('*.json'))
        if limit:
            files = files[:limit]

        processed_images = set()
        rows = []

        # If CSV exists, read processed image names
        try:
            df_existing = pd.read_csv(csv_path)
            processed_images = set(df_existing['image_name'].unique())
            print(f"[export_objects_to_csv] Found {len(processed_images)} images already in CSV.")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print("[export_objects_to_csv] No existing CSV found, starting fresh.")

        for file_path in tqdm(files, desc="Processing images"):
            image_name = file_path.stem
            if image_name in processed_images:
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            img_attrs = data.get('attributes', {})
            for frame in data.get('frames', []):
                for obj in frame.get('objects', []):
                    if 'box2d' in obj:
                        row = {
                            'image_name': image_name,
                            'category': obj.get('category'),
                            'bbox_x1': obj['box2d']['x1'],
                            'bbox_y1': obj['box2d']['y1'],
                            'bbox_x2': obj['box2d']['x2'],
                            'bbox_y2': obj['box2d']['y2'],
                            'object_id': obj.get('id'),
                            # Flatten object attributes
                            **{f"obj_attr_{k}": v for k, v in obj.get('attributes', {}).items()},
                            # Flatten image attributes
                            **{f"img_attr_{k}": v for k, v in img_attrs.items()}
                        }
                        rows.append(row)

        if rows:
            df_new = pd.DataFrame(rows)
            # Append or create CSV
            try:
                df_existing = pd.read_csv(csv_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(csv_path, index=False)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                df_new.to_csv(csv_path, index=False)
            print(f"[export_objects_to_csv] Added {len(rows)} new objects to {csv_path}")
        else:
            print("[export_objects_to_csv] No new images to process.")

    @staticmethod
    def load_or_update_csv(label_dir: str, csv_path: str, limit: Optional[int] = None):
        """
        Load the CSV if it exists and is up-to-date, otherwise create/update it.

        Args:
            label_dir (str): Directory with JSON label files.
            csv_path (str): Path to the CSV file.
            limit (Optional[int]): Max number of files to process.
        """
        parser = BDD100KBoundingBoxParser(label_dir)
        files = list(parser.label_dir.glob('*.json'))
        image_names = set(f.stem for f in files)
        try:
            df = pd.read_csv(csv_path)
            processed_images = set(df['image_name'].unique())
            if processed_images == image_names:
                print(f"[load_or_update_csv] CSV {csv_path} is up-to-date.")
                return df
            else:
                print(f"[load_or_update_csv] Updating CSV {csv_path} with new images.")
                parser.export_objects_to_csv(csv_path, limit=limit)
                return pd.read_csv(csv_path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"[load_or_update_csv] Creating CSV {csv_path}.")
            parser.export_objects_to_csv(csv_path, limit=limit)
            return pd.read_csv(csv_path)

    @staticmethod
    def load_csv(csv_path: str) -> pd.DataFrame:
        """
        Load the CSV file as a DataFrame.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame with all object and image-level info.
        """
        return pd.read_csv(csv_path)

# Example usage:
#parser = BDD100KBoundingBoxParser('./bdd100k_labels/100k/train')
#all_bboxes = parser.parse_all(limit=2)
#print(all_bboxes)
#category_counts = parser.count_categories(limit=2)
#print(category_counts)
