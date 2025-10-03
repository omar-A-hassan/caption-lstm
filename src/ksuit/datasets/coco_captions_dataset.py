import json
from pathlib import Path
from PIL import Image
import torch

from ksuit.data import Dataset
from ksuit.utils.param_checking import to_path
from ksuit.distributed import is_local_rank0, barrier


class CocoCaptionsDataset(Dataset):
    """
    COCO Captions dataset for image captioning.

    Expected structure:
        root/
            annotations/
                captions_train2017.json
                captions_val2017.json
            train2017/
                000000000009.jpg
                ...
            val2017/
                000000000139.jpg
                ...
    """

    def __init__(
        self,
        root,
        split="train",
        return_all_captions=False,
        **kwargs,
    ):
        """
        Args:
            root: Path to COCO dataset root (e.g., /kaggle/input/coco-2017-dataset/coco2017)
            split: 'train' or 'val'
            return_all_captions: If True, returns all captions for an image.
                               If False, returns one random caption per image.
            **kwargs: Additional arguments for base Dataset class
        """
        super().__init__(**kwargs)
        self.root = to_path(root)
        self.split = split
        self.return_all_captions = return_all_captions

        # Paths
        self.image_dir = self.root / f"{split}2017"
        self.ann_file = self.root / "annotations" / f"captions_{split}2017.json"

        # Load annotations
        self.logger.info(f"Loading COCO captions from {self.ann_file}")
        with open(self.ann_file, "r") as f:
            coco_data = json.load(f)

        # Build image id to info mapping
        self.images = {img["id"]: img for img in coco_data["images"]}

        # Build image id to captions mapping
        self.image_to_captions = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"]
            if image_id not in self.image_to_captions:
                self.image_to_captions[image_id] = []
            self.image_to_captions[image_id].append(caption)

        # Create list of image ids
        self.image_ids = list(self.images.keys())

        self.logger.info(
            f"Loaded {len(self.image_ids)} images with "
            f"{sum(len(caps) for caps in self.image_to_captions.values())} captions"
        )

    def getitem_x(self, idx):
        """Get image."""
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        image_path = self.image_dir / image_info["file_name"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        return image

    def getitem_caption(self, idx):
        """Get caption(s) for image."""
        image_id = self.image_ids[idx]
        captions = self.image_to_captions[image_id]

        if self.return_all_captions:
            return captions
        else:
            # Return random caption
            import random
            return random.choice(captions)

    def getitem_image_id(self, idx):
        """Get COCO image ID."""
        return self.image_ids[idx]

    def __getitem__(self, idx):
        """Return (image, caption) or (image, captions)."""
        image = self.getitem_x(idx)
        caption = self.getitem_caption(idx)
        return image, caption

    def __len__(self):
        return len(self.image_ids)
