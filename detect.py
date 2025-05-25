import torch

from ultralytics import YOLO
from torch.nn.modules.container import Sequential
torch.serialization.add_safe_globals([Sequential])
ckpt = torch.load(r'C:\Users\17412\Downloads\Strawberry_detect\models\best.pt', map_location="cpu")
model = YOLO(r'C:\Users\17412\Downloads\Strawberry_detect\models\best.pt')

results = model(r'C:\Users\17412\Downloads\Strawberry_detect\images\(150).jpg', save=True)