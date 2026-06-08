from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn as nn
import json
import os

TEST_DIR = r"D:\College Resourcez\Sem 4 Resources\Artifical Intelligence\VSC\AI Project\asl_alphabet_test"

MODEL_PATH = r"D:\College Resourcez\Sem 4 Resources\Artifical Intelligence\VSC\model_v3\model.pth"

LABELS_PATH = r"D:\College Resourcez\Sem 4 Resources\Artifical Intelligence\VSC\model_v3\labels.json"

NUM_CLASSES = 29
IMG_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# labels
with open(LABELS_PATH) as f:
    idx_to_class = {int(k): v for k, v in json.load(f).items()}

# reverse mapping
class_to_idx = {v.lower(): k for k, v in idx_to_class.items()}

# model
model = models.mobilenet_v2(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES),
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

correct = 0
total = 0

for filename in os.listdir(TEST_DIR):

    if not filename.endswith(".jpg"):
        continue

    true_label = filename.split("_test")[0].lower()

    img = Image.open(
        os.path.join(TEST_DIR, filename)
    ).convert("RGB")

    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_idx = model(x).argmax(1).item()

    pred_label = idx_to_class[pred_idx].lower()

    print(
        f"{filename:20s} | "
        f"True: {true_label:8s} | "
        f"Pred: {pred_label}"
    )

    if pred_label == true_label:
        correct += 1

    total += 1

print("\nAccuracy:", correct / total * 100)