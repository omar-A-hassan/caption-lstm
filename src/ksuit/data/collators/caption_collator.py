import torch
from .collator import Collator


class CaptionCollator(Collator):
    """
    Collator for image captioning that handles variable-length captions.

    Expects batch items to be (image, caption) tuples where:
    - image: PIL Image or Tensor
    - caption: str (single caption)

    Returns:
        dict with 'images' (batched tensors) and 'captions' (list of strings)
    """

    def __init__(self, transform=None):
        """
        Args:
            transform: Optional torchvision transform for images
        """
        super().__init__()
        self.transform = transform

    def __call__(self, batch):
        """
        Collate a batch of (image, caption) tuples.

        Args:
            batch: List of (image, caption) tuples

        Returns:
            dict with 'images' and 'captions'
        """
        images = []
        captions = []

        for image, caption in batch:
            # Apply transform if provided
            if self.transform is not None:
                image = self.transform(image)

            images.append(image)
            captions.append(caption)

        # Stack images (assumes all images are same size after transform)
        images = torch.stack(images)

        return {
            'images': images,
            'captions': captions,
        }
