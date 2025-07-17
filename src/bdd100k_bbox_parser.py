import json
from pathlib import Path
from typing import List, Dict, Any, Optional


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
        """
        Parse all JSON files in the label directory.

        Args:
            limit (Optional[int]): Maximum number of files to parse.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Mapping from filename to list of bounding boxes.
        """
        results = {}
        files = list(self.label_dir.glob('*.json'))
        if limit:
            files = files[:limit]
        for file_path in files:
            results[file_path.name] = self.parse_file(str(file_path))
        return results

    def count_categories(self, limit: Optional[int] = None) -> Dict[str, int]:
        """
        Count the number of objects per category in the dataset.

        Args:
            limit (Optional[int]): Maximum number of files to parse.

        Returns:
            Dict[str, int]: Mapping from category name to count.
        """
        category_counts = {}
        files = list(self.label_dir.glob('*.json'))
        if limit:
            files = files[:limit]
        for file_path in files:
            bboxes = self.parse_file(str(file_path))
            for bbox in bboxes:
                cat = bbox.get('category')
                if cat:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
        return category_counts

# Example usage:
#parser = BDD100KBoundingBoxParser('./bdd100k_labels/100k/train')
#all_bboxes = parser.parse_all(limit=2)
#print(all_bboxes)
#category_counts = parser.count_categories(limit=2)
#print(category_counts)
