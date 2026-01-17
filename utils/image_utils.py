from PIL import Image
import torchvision.transforms as transforms


# Image preprocessing pipeline
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def preprocess_image(image_path):
    """
    Takes image path and returns a tensor of shape (1, 3, 224, 224)
    """
    image = Image.open(image_path).convert("RGB")
    image = _transform(image)
    return image.unsqueeze(0)
